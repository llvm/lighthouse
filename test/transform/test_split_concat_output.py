# RUN: %PYTHON %s | FileCheck %s

"""Tensor-level concat splitting: structure, rejection and CPU execution."""

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.execution.runner import Runner
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.lowering_common import vectorize as vectorize_payload


SOURCE = """
#id = affine_map<(h,s,d)->(h,s,d)>
#table = affine_map<(h,s,d)->(s,d)>
func.func @rope(%x: tensor<4x1x6xf32>, %cos: tensor<4x6xf32>,
                %sin: tensor<4x6xf32>, %bias: tensor<1x4x6xf32>,
                %dest: memref<1x4x6xf32>) {
  %result = scf.forall (%i) = (0) to (4) step (2)
      shared_outs(%out = %bias) -> (tensor<1x4x6xf32>) {
    %input = tensor.extract_slice %x[%i,0,0] [2,1,6] [1,1,1]
      : tensor<4x1x6xf32> to tensor<2x1x6xf32>
    %dst = tensor.extract_slice %out[0,%i,0] [1,2,6] [1,1,1]
      : tensor<1x4x6xf32> to tensor<1x2x6xf32>
    %e = tensor.empty() : tensor<1x2x6xf32>
    %t = linalg.transpose ins(%input : tensor<2x1x6xf32>)
      outs(%e : tensor<1x2x6xf32>) permutation = [1,0,2]
    %lo = tensor.extract_slice %t[0,0,0] [1,2,3] [1,1,1]
      : tensor<1x2x6xf32> to tensor<1x2x3xf32>
    %hi = tensor.extract_slice %t[0,0,3] [1,2,3] [1,1,1]
      : tensor<1x2x6xf32> to tensor<1x2x3xf32>
    %half = tensor.empty() : tensor<1x2x3xf32>
    %neg = linalg.generic {indexing_maps=[#id,#id],
        iterator_types=["parallel","parallel","parallel"]}
      ins(%hi : tensor<1x2x3xf32>) outs(%half : tensor<1x2x3xf32>) {
    ^bb0(%a: f32, %o: f32):
      %n = arith.negf %a : f32
      linalg.yield %n : f32
    } -> tensor<1x2x3xf32>
    %rot = tensor.concat dim(2) %neg, %lo :
      (tensor<1x2x3xf32>, tensor<1x2x3xf32>) -> tensor<1x2x6xf32>
    %c = tensor.extract_slice %cos[%i,0] [2,6] [1,1]
      : tensor<4x6xf32> to tensor<2x6xf32>
    %s = tensor.extract_slice %sin[%i,0] [2,6] [1,1]
      : tensor<4x6xf32> to tensor<2x6xf32>
    %sum = linalg.generic {indexing_maps=[#id,#id,#table,#table,#id],
        iterator_types=["parallel","parallel","parallel"]}
      ins(%t, %rot, %c, %s : tensor<1x2x6xf32>, tensor<1x2x6xf32>,
                            tensor<2x6xf32>, tensor<2x6xf32>)
      outs(%dst : tensor<1x2x6xf32>) {
    ^bb0(%a: f32, %r: f32, %cv: f32, %sv: f32, %o: f32):
      %ac = arith.mulf %a, %cv : f32
      %rs = arith.mulf %r, %sv : f32
      %v = arith.addf %ac, %rs : f32
      %acc = arith.addf %v, %o : f32
      linalg.yield %acc : f32
    } -> tensor<1x2x6xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %sum into %out[0,%i,0] [1,2,6] [1,1,1]
        : tensor<1x2x6xf32> into tensor<1x4x6xf32>
    }
  }
  bufferization.materialize_in_destination %result in restrict writable %dest
    : (tensor<1x4x6xf32>, memref<1x4x6xf32>) -> ()
  return
}
"""

PERMUTED = """
#id = affine_map<(i,j)->(i,j)>
#perm = affine_map<(i,j)->(j,i)>
func.func @rope(%a: tensor<2x3xf32>, %b: tensor<2x5xf32>,
                %dest: memref<8x2xf32>) {
  %e = tensor.empty() : tensor<8x2xf32>
  %result = scf.forall (%i) in (1) shared_outs(%out = %e)
      -> (tensor<8x2xf32>) {
    %concat = tensor.concat dim(1) %a, %b :
      (tensor<2x3xf32>, tensor<2x5xf32>) -> tensor<2x8xf32>
    %v = linalg.generic {indexing_maps=[#id,#perm],
        iterator_types=["parallel","parallel"]}
      ins(%concat : tensor<2x8xf32>) outs(%out : tensor<8x2xf32>) {
    ^bb0(%x: f32, %o: f32):
      %n = arith.negf %x : f32
      linalg.yield %n : f32
    } -> tensor<8x2xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %v into %out[0,0] [8,2] [1,1]
        : tensor<8x2xf32> into tensor<8x2xf32>
    }
  }
  bufferization.materialize_in_destination %result in restrict writable %dest
    : (tensor<8x2xf32>, memref<8x2xf32>) -> ()
  return
}
"""

RESHAPED = SOURCE.replace("%rot = tensor.concat", "%raw_rot = tensor.concat").replace(
    "    %c = tensor.extract_slice",
    """    %flat_rot = tensor.collapse_shape %raw_rot [[0,1],[2]]
      : tensor<1x2x6xf32> into tensor<2x6xf32>
    %rot = tensor.expand_shape %flat_rot [[0,1],[2]] output_shape [1,2,6]
      : tensor<2x6xf32> into tensor<1x2x6xf32>
    %c = tensor.extract_slice""",
)


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(source=SOURCE, *, split=True, vectorize=False, target_forall=False):
    module = ir.Module.parse(source)
    with schedule_boilerplate() as (schedule, sequence):
        func = lh_transform.match_op(sequence.bodyTarget, "func.func").results[0]
        if split:
            if target_forall:
                transform_ext.split_concat_output(
                    lh_transform.match_op(func, "scf.forall").results[0]
                )
            else:
                func = transform_ext.split_concat_output(func)
        lh_transform.cleanup(func)
        if vectorize:
            vectorize_payload(sequence.bodyTarget, payload_func=func)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure():
    module = apply()
    assert len(operations(module, "scf.forall")) == 1
    assert len(operations(module, "tensor.parallel_insert_slice")) == 1
    inserts = operations(module, "tensor.insert_slice")
    assert len(inserts) == 2
    assert [list(op.static_offsets)[-1] for op in inserts] == [0, 3]
    assert all(list(op.static_sizes) == [1, 2, 3] for op in inserts)
    assert not operations(module, "tensor.concat")
    for op in operations(module, "tensor.extract_slice"):
        assert list(op.static_sizes)[-1] == 3 or isinstance(op.source.owner, ir.Block)
        assert op.source.owner not in [
            t.operation for t in operations(module, "linalg.transpose")
        ]
    assert str(apply(str(module))) == str(module)
    assert str(apply(target_forall=True)) == str(module)
    for source in (SOURCE, RESHAPED):
        vectorized = apply(source, vectorize=True)
        for name in ("vector.extract_strided_slice", "vector.insert_strided_slice"):
            assert not operations(vectorized, name)
    print("One forall, disjoint half stores, direct half loads, no vector slices")


# CHECK: One forall, disjoint half stores, direct half loads, no vector slices


def test_permuted_output():
    module = apply(PERMUTED)
    inserts = operations(module, "tensor.insert_slice")
    assert [list(op.static_offsets) for op in inserts] == [[0, 0], [3, 0]]
    assert [list(op.static_sizes) for op in inserts] == [[3, 2], [5, 2]]
    assert not operations(module, "tensor.concat")
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 3)).astype(np.float32)
    b = rng.normal(size=(2, 5)).astype(np.float32)
    expected = -np.concatenate((a, b), axis=1).T
    for split in (False, True):
        execute(apply(PERMUTED, split=split), [a, b], expected)
    print("Unequal concat pieces and permuted output maps are preserved")


# CHECK: Unequal concat pieces and permuted output maps are preserved


def test_rejections():
    indexed = SOURCE.replace(
        "%n = arith.negf %a : f32",
        "%idx = linalg.index 2 : index\n"
        "%j = arith.index_cast %idx : index to i64\n"
        "%f = arith.sitofp %j : i64 to f32\n"
        "%n = arith.addf %a, %f : f32",
    )
    strided = SOURCE.replace(
        "%lo = tensor.extract_slice %t[0,0,0] [1,2,3] [1,1,1]",
        "%lo = tensor.extract_slice %t[0,0,0] [1,2,3] [1,1,2]",
    )
    dynamic = SOURCE.replace("tensor<4x1x6xf32>", "tensor<?x1x6xf32>")
    no_concat = SOURCE.replace("ins(%t, %rot, %c, %s", "ins(%t, %t, %c, %s")
    for source in (indexed, strided, dynamic, no_concat):
        assert str(apply(source)) == str(apply(source, split=False))
    print("Unsupported plans leave the entire output unchanged")


# CHECK: Unsupported plans leave the entire output unchanged


def execute(module, inputs, expected):
    PassManager.parse(
        "builtin.module(empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "allow-return-allocs-from-loops "
        "function-boundary-type-conversion=identity-layout-map},"
        "scf-forall-to-for,buffer-deallocation-pipeline,"
        "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf,convert-math-to-llvm)"
    ).run(module.operation)
    Runner.make_function_callable(module, "rope")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    output = np.empty(expected.shape, dtype=np.float32)
    Runner(module).execute("rope", [*(x.copy() for x in inputs), output])
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)


def test_numerics():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 1, 6)).astype(np.float32)
    cos = rng.normal(size=(4, 6)).astype(np.float32)
    sin = rng.normal(size=(4, 6)).astype(np.float32)
    bias = rng.normal(size=(1, 4, 6)).astype(np.float32)
    transposed = x.transpose(1, 0, 2)
    rotated = np.concatenate((-transposed[..., 3:], transposed[..., :3]), axis=-1)
    expected = transposed * cos + rotated * sin + bias
    for split in (False, True):
        execute(apply(split=split), [x, cos, sin, bias], expected)
        reshaped = apply(RESHAPED, split=split)
        if split:
            assert not operations(reshaped, "tensor.concat")
        execute(reshaped, [x, cos, sin, bias], expected)
    rounded = SOURCE.replace(
        "%ac = arith.mulf %a, %cv : f32",
        "%ac_full = arith.mulf %a, %cv : f32\n"
        "%ac_half = arith.truncf %ac_full : f32 to f16\n"
        "%ac = arith.extf %ac_half : f16 to f32",
    ).replace(
        "%rs = arith.mulf %r, %sv : f32",
        "%rs_full = arith.mulf %r, %sv : f32\n"
        "%rs_half = arith.truncf %rs_full : f32 to f16\n"
        "%rs = arith.extf %rs_half : f16 to f32",
    )
    expected = (
        (transposed * cos).astype(np.float16).astype(np.float32)
        + (rotated * sin).astype(np.float16).astype(np.float32)
        + bias
    )
    for split in (False, True):
        execute(apply(rounded, split=split), [x, cos, sin, bias], expected)
    print("Split and original RoPE match NumPy, including broadcast and DPS reads")


# CHECK: Split and original RoPE match NumPy, including broadcast and DPS reads


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        test_structure()
        test_permuted_output()
        test_rejections()
        test_numerics()

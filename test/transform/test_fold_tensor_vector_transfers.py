# RUN: %PYTHON %s | FileCheck %s

"""Transfer forwarding legality and executable tensor/vector round-trip tests."""

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.execution.runner import Runner
from lighthouse.schedule import schedule_boilerplate


SOURCE = """
func.func @rope(%input: memref<2x8xf32>, %output: memref<2x2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %zero = arith.constant 0.0 : f32
  %v = vector.transfer_read %input[%c0,%c0], %zero {in_bounds=[true,true]}
    : memref<2x8xf32>, vector<2x8xf32>
  %small = arith.truncf %v : vector<2x8xf32> to vector<2x8xf16>
  %rounded = arith.extf %small : vector<2x8xf16> to vector<2x8xf32>
  %e = tensor.empty() : tensor<2x8xf32>
  %p = vector.transfer_write %rounded, %e[%c0,%c0] {in_bounds=[true,true]}
    : vector<2x8xf32>, tensor<2x8xf32>
  %expanded = tensor.expand_shape %p [[0,1],[2,3]] output_shape [1,2,2,4]
    : tensor<2x8xf32> into tensor<1x2x2x4xf32>
  %r = vector.transfer_read %expanded[%c0,%c0,%c0,%c0], %zero
    {in_bounds=[true,true,true]} : tensor<1x2x2x4xf32>, vector<2x2x4xf32>
  %t = vector.transpose %r, [1,0,2] : vector<2x2x4xf32> to vector<2x2x4xf32>
  %te = tensor.empty() : tensor<1x2x2x4xf32>
  %tw = vector.transfer_write %t, %te[%c0,%c0,%c0,%c0]
    {in_bounds=[true,true,true]} : vector<2x2x4xf32>, tensor<1x2x2x4xf32>
  %half = tensor.extract_slice %tw[0,0,0,0] [1,2,2,2] [1,1,1,1]
    : tensor<1x2x2x4xf32> to tensor<1x2x2x2xf32>
  %second = vector.transfer_read %tw[%c0,%c0,%c0,%c2], %zero
    {in_bounds=[true,true,true]} : tensor<1x2x2x4xf32>, vector<2x2x2xf32>
  %neg = arith.negf %second : vector<2x2x2xf32>
  %he = tensor.empty() : tensor<1x2x2x2xf32>
  %hw = vector.transfer_write %neg, %he[%c0,%c0,%c0,%c0]
    {in_bounds=[true,true,true]} : vector<2x2x2xf32>, tensor<1x2x2x2xf32>
  %concat = tensor.concat dim(3) %hw, %half
    : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x4xf32>
  %result = vector.transfer_read %concat[%c0,%c0,%c0,%c0], %zero
    {in_bounds=[true,true,true]} : tensor<1x2x2x4xf32>, vector<2x2x4xf32>
  vector.transfer_write %result, %output[%c0,%c0,%c0] {in_bounds=[true,true,true]}
    : vector<2x2x4xf32>, memref<2x2x4xf32>
  return
}
"""

SUBSET = """
func.func @subset(%v: vector<2x4xf32>, %base: tensor<4x8xf32>)
    -> vector<1x2xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %zero = arith.constant 0.0 : f32
  %w = vector.transfer_write %v, %base[%c1,%c2] {in_bounds=[true,true]}
    : vector<2x4xf32>, tensor<4x8xf32>
  %r = vector.transfer_read %w[%c2,%c3], %zero {in_bounds=[true,true]}
    : tensor<4x8xf32>, vector<1x2xf32>
  return %r : vector<1x2xf32>
}
"""

RESHAPED_WRITE = """
func.func @reshape_write(%v: vector<4x1x8xf32>, %base: tensor<1x4x1x8xf32>)
    -> tensor<4x8xf32> {
  %c0 = arith.constant 0 : index
  %w = vector.transfer_write %v, %base[%c0,%c0,%c0,%c0]
    {in_bounds=[true,true,true]} : vector<4x1x8xf32>, tensor<1x4x1x8xf32>
  %c = tensor.collapse_shape %w [[0,1],[2,3]]
    : tensor<1x4x1x8xf32> into tensor<4x8xf32>
  return %c : tensor<4x8xf32>
}
"""


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(source):
    module = ir.Module.parse(source)
    assert module.operation.verify()
    with schedule_boilerplate() as (schedule, sequence):
        target = transform_ext.fold_tensor_vector_transfers(sequence.bodyTarget)
        lh_transform.cleanup(target)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure():
    module = apply(SOURCE)
    assert "tensor." not in str(module)
    assert len(operations(module, "vector.transfer_read")) == 1
    assert len(operations(module, "vector.transfer_write")) == 1
    assert len(operations(module, "arith.truncf")) == 1
    assert len(operations(module, "arith.extf")) == 1
    assert operations(module, "vector.extract_strided_slice")
    assert operations(module, "vector.insert_strided_slice")
    assert str(apply(str(module))) == str(module)
    subset = apply(SUBSET)
    assert not operations(subset, "vector.transfer_read")
    (extract,) = operations(subset, "vector.extract_strided_slice")
    assert (
        list(ir.ArrayAttr(extract.attributes["offsets"]))
        == [ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)] * 2
    )
    collapsed = """
func.func @collapse(%v: vector<2x2x4xf32>) -> vector<2x8xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<2x2x4xf32>
  %w = vector.transfer_write %v, %e[%c0,%c0,%c0] {in_bounds=[true,true,true]}
    : vector<2x2x4xf32>, tensor<2x2x4xf32>
  %c = tensor.collapse_shape %w [[0],[1,2]]
    : tensor<2x2x4xf32> into tensor<2x8xf32>
  %r = vector.transfer_read %c[%c0,%c0], %zero {in_bounds=[true,true]}
    : tensor<2x8xf32>, vector<2x8xf32>
  return %r : vector<2x8xf32>
}
"""
    assert "tensor." not in str(apply(collapsed))
    reshaped = apply(RESHAPED_WRITE)
    assert not operations(reshaped, "tensor.collapse_shape")
    (write,) = operations(reshaped, "vector.transfer_write")
    assert list(write.opview.valueToStore.type.shape) == [4, 8]
    assert list(write.opview.base.type.shape) == [4, 8]
    print("Reshape, subset and concat transfers stay in vector SSA")


# CHECK: Reshape, subset and concat transfers stay in vector SSA


def test_rejections():
    partial_write = RESHAPED_WRITE.replace("vector<4x1x8xf32>", "vector<2x1x8xf32>")
    assert operations(apply(partial_write), "tensor.collapse_shape")
    masked_reshape = RESHAPED_WRITE.replace(
        "  %w =",
        "  %mask = arith.constant dense<false> : vector<4x1x8xi1>\n  %w =",
    ).replace("%base[%c0,%c0,%c0,%c0]", "%base[%c0,%c0,%c0,%c0], %mask")
    assert operations(apply(masked_reshape), "tensor.collapse_shape")
    outside = SUBSET.replace("%w[%c2,%c3]", "%w[%c3,%c3]")
    assert len(operations(apply(outside), "vector.transfer_read")) == 1
    dynamic = SUBSET.replace(
        "%base: tensor<4x8xf32>", "%base: tensor<4x8xf32>, %index: index"
    ).replace("%w[%c2,%c3]", "%w[%index,%c3]")
    assert len(operations(apply(dynamic), "vector.transfer_read")) == 1
    masked = SUBSET.replace(
        "%zero = arith.constant 0.0 : f32",
        "%mask = arith.constant dense<[[true,false]]> : vector<1x2xi1>\n"
        "  %zero = arith.constant 0.0 : f32",
    ).replace("%w[%c2,%c3], %zero", "%w[%c2,%c3], %zero, %mask")
    assert len(operations(apply(masked), "vector.transfer_read")) == 1
    masked_write = SUBSET.replace(
        "%zero = arith.constant 0.0 : f32",
        "%mask = arith.constant dense<false> : vector<2x4xi1>\n"
        "  %zero = arith.constant 0.0 : f32",
    ).replace("%base[%c1,%c2] {", "%base[%c1,%c2], %mask {")
    assert len(operations(apply(masked_write), "vector.transfer_read")) == 1
    permuted = SUBSET.replace(
        "%w[%c2,%c3], %zero {in_bounds=[true,true]}",
        "%w[%c2,%c3], %zero {in_bounds=[true,true],"
        "permutation_map=affine_map<(i,j)->(j,i)>}",
    )
    assert len(operations(apply(permuted), "vector.transfer_read")) == 1
    scalable = SUBSET.replace("vector<2x4xf32>", "vector<[2]x4xf32>")
    assert len(operations(apply(scalable), "vector.transfer_read")) == 1
    out_of_bounds = SUBSET.replace("vector<1x2xf32>", "vector<1x8xf32>").replace(
        "%w[%c2,%c3], %zero {in_bounds=[true,true]}",
        "%w[%c2,%c3], %zero {in_bounds=[true,false]}",
    )
    assert len(operations(apply(out_of_bounds), "vector.transfer_read")) == 1
    print("Uncovered, dynamic, masked, permuted and out-of-bounds reads are preserved")


# CHECK: Uncovered, dynamic, masked, permuted and out-of-bounds reads are preserved


def test_native_subset_folding():
    module = apply("""
func.func @slice(%input: tensor<1x4x8xf32>, %index: index) -> vector<2x2xf32> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  %slice = tensor.extract_slice %input[0,0,2] [1,4,4] [1,1,1]
    : tensor<1x4x8xf32> to tensor<4x4xf32>
  %r = vector.transfer_read %slice[%index,%c0], %zero {in_bounds=[true,true]}
    : tensor<4x4xf32>, vector<2x2xf32>
  return %r : vector<2x2xf32>
}
""")
    assert not operations(module, "tensor.extract_slice")
    (read,) = operations(module, "vector.transfer_read")
    assert list(read.opview.base.type.shape) == [1, 4, 8]
    assert len(read.opview.indices) == 3
    print("Native subset folding handles rank reduction and dynamic read indices")


# CHECK: Native subset folding handles rank reduction and dynamic read indices


def test_handle_invalidation():
    module = ir.Module.parse(SUBSET)
    with schedule_boilerplate() as (schedule, sequence):
        reads = lh_transform.match_op(sequence.bodyTarget, "vector.transfer_read")
        transform_ext.fold_tensor_vector_transfers(sequence.bodyTarget)
        transform.annotate(reads, "test.stale")
        transform.yield_([])
    try:
        schedule.body.operations[0].apply(module.operation)
    except ValueError as error:
        assert "invalidated" in str(error), str(error)
    else:
        raise AssertionError("Rewriting must invalidate nested handles")
    print("Transfer folding invalidates handles to rewritten nested operations")


# CHECK: Transfer folding invalidates handles to rewritten nested operations


def test_numerics():
    module = apply(SOURCE)
    PassManager.parse(
        "builtin.module(convert-vector-to-scf{full-unroll},"
        "convert-vector-to-llvm,expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf)"
    ).run(module.operation)
    Runner.make_function_callable(module, "rope")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    runner = Runner(module)
    rng = np.random.default_rng(42)
    inputs = [
        rng.normal(size=(2, 8)).astype(np.float32),
        np.array(
            [0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, 0.00001, -0.00001] * 2,
            dtype=np.float32,
        ).reshape(2, 8),
    ]
    for data in inputs:
        rounded = data.astype(np.float16).astype(np.float32)
        heads = rounded.reshape(2, 2, 4).transpose(1, 0, 2)
        expected = np.concatenate((-heads[..., 2:], heads[..., :2]), axis=-1)
        actual = np.empty((2, 2, 4), dtype=np.float32)
        runner.execute("rope", [data, actual])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    print("Forwarding preserves rounded values, NaNs, infinities and signed zeros")


# CHECK: Forwarding preserves rounded values, NaNs, infinities and signed zeros


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    test_structure()
    test_rejections()
    test_native_subset_folding()
    test_handle_invalidation()
    test_numerics()

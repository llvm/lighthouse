# RUN: %PYTHON %s | FileCheck %s

"""Structural, legality and executable tests for forward epilogue DAG fusion."""

import numpy as np
from mlir import ir
from mlir.dialects import transform
from mlir.passmanager import PassManager

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.dialects.transform.transform_ext.ops.fuse_forall_epilogue import (
    _bounded_offset,
)
from lighthouse.execution.runner import Runner
from lighthouse.schedule import schedule_boilerplate


SOURCE = """
#id = affine_map<(b,h,s,d)->(b,h,s,d)>
#table = affine_map<(b,h,s,d)->(s,d)>
func.func @projection(%x: tensor<8x3xf32>, %w: tensor<3x16xf32>,
                     %cos: tensor<4x4xf32>, %sin: tensor<4x4xf32>)
                     -> tensor<2x4x4x4xf32> {
  %zero = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<8x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%e : tensor<8x16xf32>) -> tensor<8x16xf32>
  %p = linalg.matmul ins(%x, %w : tensor<8x3xf32>, tensor<3x16xf32>)
                     outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  %heads = tensor.expand_shape %p [[0,1],[2,3]] output_shape [2,4,4,4]
    : tensor<8x16xf32> into tensor<2x4x4x4xf32>
  %out = tensor.empty() : tensor<2x4x4x4xf32>
  %t = linalg.transpose ins(%heads : tensor<2x4x4x4xf32>)
    outs(%out : tensor<2x4x4x4xf32>) permutation = [0,2,1,3]
  %first = tensor.extract_slice %t[0,0,0,0] [2,4,4,2] [1,1,1,1]
    : tensor<2x4x4x4xf32> to tensor<2x4x4x2xf32>
  %second = tensor.extract_slice %t[0,0,0,2] [2,4,4,2] [1,1,1,1]
    : tensor<2x4x4x4xf32> to tensor<2x4x4x2xf32>
  %half = tensor.empty() : tensor<2x4x4x2xf32>
  %neg = linalg.generic {indexing_maps=[#id,#id],
      iterator_types=["parallel","parallel","parallel","parallel"]}
    ins(%second : tensor<2x4x4x2xf32>) outs(%half : tensor<2x4x4x2xf32>) {
  ^bb0(%a: f32, %o: f32):
    %n = arith.negf %a : f32
    linalg.yield %n : f32
  } -> tensor<2x4x4x2xf32>
  %rot = tensor.concat dim(3) %neg, %first :
    (tensor<2x4x4x2xf32>, tensor<2x4x4x2xf32>) -> tensor<2x4x4x4xf32>
  %result = linalg.generic {indexing_maps=[#id,#id,#table,#table,#id],
      iterator_types=["parallel","parallel","parallel","parallel"]}
    ins(%t, %rot, %cos, %sin : tensor<2x4x4x4xf32>, tensor<2x4x4x4xf32>,
                             tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<2x4x4x4xf32>) {
  ^bb0(%a: f32, %r: f32, %c: f32, %s: f32, %o: f32):
    %ac = arith.mulf %a, %c : f32
    %rs = arith.mulf %r, %s : f32
    %sum = arith.addf %ac, %rs : f32
    linalg.yield %sum : f32
  } -> tensor<2x4x4x4xf32>
  return %result : tensor<2x4x4x4xf32>
}
"""


def schedule(tile=(2, 8), *, fuse=True):
    with schedule_boilerplate() as (module, sequence):
        matmul = lh_transform.match_op(sequence.bodyTarget, "linalg.matmul").results[0]
        _, [loop], _ = lh_transform.tile(
            matmul,
            tile_sizes=list(tile),
            use_forall=True,
            fuse_producers=True,
            apply_cleanup=False,
        )
        transform.annotate(loop, "test.marker")
        if fuse:
            loop = transform_ext.fuse_forall_epilogue(loop)
        lh_transform.cleanup(sequence.bodyTarget)
        transform.yield_([])
    return module


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def apply(source=SOURCE, tile=(2, 8), *, fuse=True):
    module = ir.Module.parse(source)
    transforms = schedule(tile, fuse=fuse)
    transforms.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure():
    module = apply()
    (loop,) = operations(module, "scf.forall")
    (matmul,) = operations(module, "linalg.matmul")
    assert matmul.operation.parent == loop.operation
    assert list(matmul.results[0].type.shape) == [2, 8]
    assert list(loop.results[0].type.shape) == [2, 4, 4, 4]
    assert "test.marker" in loop.attributes
    for name in (
        "tensor.expand_shape",
        "linalg.transpose",
        "tensor.concat",
        "arith.negf",
        "arith.mulf",
        "arith.addf",
    ):
        for op in operations(module, name):
            parent = op.operation.parent
            while parent != loop.operation:
                assert parent is not None, f"{name} escaped workgroup"
                parent = parent.parent
    (transpose,) = operations(module, "linalg.transpose")
    result = operations(module, "linalg.generic")[-1]
    assert result.operands[0] == transpose.results[0]
    (concat,) = operations(module, "tensor.concat")
    assert result.operands[1] == concat.result
    before = str(module)
    with schedule_boilerplate() as (again, sequence):
        transform_ext.fuse_forall_epilogue(
            lh_transform.match_op(sequence.bodyTarget, "scf.forall").results[0]
        )
        transform.yield_([])
    again.body.operations[0].apply(module.operation)
    assert str(module) == before
    print("Epilogue DAG is local, projection is shared, fusion is idempotent")


# CHECK: Epilogue DAG is local, projection is shared, fusion is idempotent


def test_rejections():
    for tile in ((2, 2), (3, 8), (2, 3)):
        # Half-head tiles cannot own the shuffle; remainder and nonrectangular
        # reshape tiles are deliberately outside the supported subset.
        unchanged = apply(tile=tile, fuse=False)
        attempted = apply(tile=tile)
        assert str(unchanged) == str(attempted), tile
    indexed = SOURCE.replace(
        "%n = arith.negf %a : f32",
        "%i = linalg.index 0 : index\n"
        "%f = arith.index_cast %i : index to i64\n"
        "%v = arith.sitofp %f : i64 to f32\n"
        "%n = arith.addf %a, %v : f32",
    )
    assert str(apply(indexed)) == str(apply(indexed, fuse=False))
    prefix = (
        SOURCE[: SOURCE.index("  %heads =")]
        .replace("-> tensor<2x4x4x4xf32>", "-> tensor<2x4x3x4xf32>")
        .replace("3x16", "3x12")
        .replace("8x16", "8x12")
    )
    nonrectangular = (
        prefix
        + """
  %heads = tensor.expand_shape %p [[0,1],[2,3]] output_shape [2,4,3,4]
    : tensor<8x12xf32> into tensor<2x4x3x4xf32>
  return %heads : tensor<2x4x3x4xf32>
}
"""
    )
    assert str(apply(nonrectangular, (2, 6))) == str(
        apply(nonrectangular, (2, 6), fuse=False)
    )
    noncontiguous = SOURCE.replace(
        "                     -> tensor<2x4x4x4xf32>",
        "                     -> tensor<2x16x4xf32>",
    ).replace(
        "return %result : tensor<2x4x4x4xf32>",
        "%flat = tensor.collapse_shape %result [[0],[1,2],[3]] "
        ": tensor<2x4x4x4xf32> into tensor<2x16x4xf32>\n"
        "return %flat : tensor<2x16x4xf32>",
    )
    assert str(apply(noncontiguous)) == str(apply(noncontiguous, fuse=False))
    print("Unsupported tile images and indexed bodies are unchanged")


# CHECK: Unsupported tile images and indexed bodies are unchanged


def test_external_users():
    source = (
        SOURCE.replace(
            "  return %result",
            "  func.call @observe(%p) : (tensor<8x16xf32>) -> ()\n  return %result",
        )
        + "\nfunc.func private @observe(tensor<8x16xf32>)"
    )
    module = apply(source)
    (loop,) = operations(module, "scf.forall")
    assert len(loop.results) == 2
    (call,) = operations(module, "func.call")
    assert call.operands[0] == loop.results[0]
    assert len(operations(module, "linalg.matmul")) == 1
    early_observer = (
        SOURCE.replace(
            "  %heads =",
            "  func.call @observe(%p) : (tensor<8x16xf32>) -> ()\n  %heads =",
        )
        + "\nfunc.func private @observe(tensor<8x16xf32>)"
    )
    assert str(apply(early_observer)) == str(apply(early_observer, fuse=False))
    effect = (
        SOURCE.replace(
            "  %heads =",
            "  func.call @effect() : () -> ()\n  %heads =",
        )
        + "\nfunc.func private @effect()"
    )
    assert str(apply(effect)) == str(apply(effect, fuse=False))
    print("External projection users are preserved without recomputation")


# CHECK: External projection users are preserved without recomputation


def execute(source, tile, fuse, inputs, expected, *, result="result"):
    shape = "x".join(str(s) for s in expected.shape) + "xf32"
    source = source.replace(
        ")\n                     -> tensor<" + shape + "> {",
        ", %dest: memref<" + shape + ">) {",
    ).replace(
        f"return %{result} : tensor<{shape}>",
        f"bufferization.materialize_in_destination %{result} in restrict "
        f"writable %dest : (tensor<{shape}>, memref<{shape}>) -> ()\nreturn",
    )
    module = apply(source, tile, fuse=fuse)
    PassManager.parse(
        "builtin.module(empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "allow-return-allocs-from-loops "
        "function-boundary-type-conversion=identity-layout-map},"
        "scf-forall-to-for,buffer-deallocation-pipeline,"
        "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
        "convert-scf-to-cf,convert-math-to-llvm)"
    ).run(module.operation)
    Runner.make_function_callable(module, "projection")
    PassManager.parse(
        "builtin.module(finalize-memref-to-llvm,convert-to-llvm,"
        "reconcile-unrealized-casts)"
    ).run(module.operation)
    runner = Runner(module)
    output = np.empty(expected.shape, dtype=np.float32)
    runner.execute("projection", [*inputs, output])
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)


def test_numerics():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(8, 3)).astype(np.float32)
    w = rng.normal(size=(3, 16)).astype(np.float32)
    cos = rng.normal(size=(4, 4)).astype(np.float32)
    sin = rng.normal(size=(4, 4)).astype(np.float32)
    projected = (x @ w).reshape(2, 4, 4, 4).transpose(0, 2, 1, 3)
    rotated = np.concatenate((-projected[..., 2:], projected[..., :2]), axis=-1)
    expected = projected * cos + rotated * sin
    for tile in ((2, 8), (4, 4), (8, 8), (2, 16), (16, 32)):
        for fuse in (False, True):
            execute(SOURCE, tile, fuse, [x, w, cos, sin], expected)
    collapsed = SOURCE.replace(
        "                     -> tensor<2x4x4x4xf32>",
        "                     -> tensor<2x4x16xf32>",
    ).replace(
        "return %result : tensor<2x4x4x4xf32>",
        "%flat = tensor.collapse_shape %result [[0],[1],[2,3]] "
        ": tensor<2x4x4x4xf32> into tensor<2x4x16xf32>\n"
        "return %flat : tensor<2x4x16xf32>",
    )
    execute(
        collapsed,
        (2, 8),
        True,
        [x, w, cos, sin],
        expected.reshape(2, 4, 16),
        result="flat",
    )
    bias = rng.normal(size=expected.shape).astype(np.float32)
    accumulated = (
        SOURCE.replace(
            "%sin: tensor<4x4xf32>)",
            "%sin: tensor<4x4xf32>, %bias: tensor<2x4x4x4xf32>)",
        )
        .replace(
            "    outs(%out : tensor<2x4x4x4xf32>) {",
            "    outs(%bias : tensor<2x4x4x4xf32>) {",
        )
        .replace(
            "%sum = arith.addf %ac, %rs : f32",
            "%v = arith.addf %ac, %rs : f32\n    %sum = arith.addf %v, %o : f32",
        )
    )
    execute(accumulated, (2, 8), True, [x, w, cos, sin, bias], expected + bias)
    rounded = SOURCE.replace(
        "  %heads = tensor.expand_shape %p",
        """
  %small_empty = tensor.empty() : tensor<8x16xf16>
  %small = linalg.generic {
      indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
      iterator_types=["parallel","parallel"]}
    ins(%p : tensor<8x16xf32>) outs(%small_empty : tensor<8x16xf16>) {
  ^bb0(%a: f32, %o: f16):
    %v = arith.truncf %a : f32 to f16
    linalg.yield %v : f16
  } -> tensor<8x16xf16>
  %rounded = linalg.generic {
      indexing_maps=[affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>],
      iterator_types=["parallel","parallel"]}
    ins(%small : tensor<8x16xf16>) outs(%e : tensor<8x16xf32>) {
  ^bb0(%a: f16, %o: f32):
    %v = arith.extf %a : f16 to f32
    linalg.yield %v : f32
  } -> tensor<8x16xf32>
  %heads = tensor.expand_shape %rounded""",
    )
    quantized = projected.astype(np.float16).astype(np.float32)
    rotated = np.concatenate((-quantized[..., 2:], quantized[..., :2]), axis=-1)
    execute(rounded, (2, 8), True, [x, w, cos, sin], quantized * cos + rotated * sin)
    print("Fused and unfused projection/RoPE match NumPy across tile shapes")


# CHECK: Fused and unfused projection/RoPE match NumPy across tile shapes


def test_rank_reduced_nested_consumer():
    module = ir.Module.parse("""
func.func @heads(%x: tensor<2x2x4x4xf32>) -> tensor<8x8xf32> {
  %empty = tensor.empty() : tensor<2x2x4x4xf32>
  %heads = scf.forall (%b, %h, %s) = (0,0,0) to (2,2,4) step (1,1,2)
      shared_outs(%dest = %empty) -> tensor<2x2x4x4xf32> {
    %tile = tensor.extract_slice %x[%b,%h,%s,0] [1,1,2,4] [1,1,1,1]
      : tensor<2x2x4x4xf32> to tensor<2x4xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %dest[%b,%h,%s,0] [1,1,2,4] [1,1,1,1]
        : tensor<2x4xf32> into tensor<2x2x4x4xf32>
    }
  }
  %out = tensor.empty() : tensor<2x4x2x4xf32>
  %t = linalg.transpose ins(%heads : tensor<2x2x4x4xf32>)
      outs(%out : tensor<2x4x2x4xf32>) permutation = [0,2,1,3]
  %flat = tensor.collapse_shape %t [[0,1],[2,3]]
      : tensor<2x4x2x4xf32> into tensor<8x8xf32>
  %e = tensor.empty() : tensor<8x8xf32>
  %result = scf.forall (%m) in (2) shared_outs(%dest = %e) -> tensor<8x8xf32> {
    %offset = affine.apply affine_map<(i)->(i*4)>(%m)
    %tile = tensor.extract_slice %flat[%offset,0] [4,8] [1,1]
      : tensor<8x8xf32> to tensor<4x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %dest[%offset,0] [4,8] [1,1]
        : tensor<4x8xf32> into tensor<8x8xf32>
    }
  }
  return %result : tensor<8x8xf32>
}
""")
    with schedule_boilerplate() as (schedule, sequence):
        loops = lh_transform.match_op(sequence.bodyTarget, "scf.forall")
        loop = transform_ext.extract_handle(loops, 0)
        transform_ext.fuse_forall_epilogue(loop)
        transform.yield_([])
    schedule.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    loops = operations(module, "scf.forall")
    assert len(loops) == 2
    assert list(loops[0].result.type.shape) == [8, 8]
    assert operations(loops[0], "linalg.transpose")
    assert operations(loops[1], "tensor.extract_slice")[0].source == loops[0].result
    print("Rank-reduced tiles fuse through head merging into nested consumers")


# CHECK: Rank-reduced tiles fuse through head merging into nested consumers


def test_grouped_head_offsets():
    kv_head = ir.AffineExpr.get_dim(0)
    group = ir.AffineExpr.get_dim(1)
    flattened = kv_head * 4 + group
    bounds = [(0, 7), (0, 3)]
    restored_head = (
        ir.AffineExpr.get_floor_div(flattened, ir.AffineExpr.get_constant(4)) % 8
    )
    restored_group = flattened % 4
    assert _bounded_offset(restored_head, bounds) == (kv_head, 0, 7)
    assert _bounded_offset(restored_group, bounds) == (group, 0, 3)
    # Do not discard mod when the IV crosses more than one group.
    assert _bounded_offset(group % 4, [(0, 7), (0, 7)])[0] == group % 4
    print("Grouped-head collapse/expand offsets agree within forall bounds")


# CHECK: Grouped-head collapse/expand offsets agree within forall bounds


with ir.Context(), ir.Location.unknown():
    lh_dialects.register_and_load()
    test_structure()
    test_rejections()
    test_external_users()
    test_numerics()
    test_rank_reduced_nested_consumer()
    test_grouped_head_offsets()

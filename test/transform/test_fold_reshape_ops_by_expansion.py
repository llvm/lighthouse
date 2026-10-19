# RUN: %PYTHON %s | FileCheck %s

"""Static reshape folding preserves bodies, indexing and shared results."""

import numpy as np
from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.utils.mlir import indexing_maps
from test_attention_mask_numerics import compile_payload
from test_sink_normalization_producers import reshaped_source


COLLAPSE = """
func.func @contraction(%x: tensor<2x3x4xf32>, %w: tensor<4x5xf32>) -> tensor<2x3x5xf32> {
  %zero = arith.constant 0.0 : f32
  %empty_x = tensor.empty() : tensor<2x3x4xf32>
  %neg = linalg.generic {
    indexing_maps = [affine_map<(a,b,c)->(a,b,c)>, affine_map<(a,b,c)->(a,b,c)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%x : tensor<2x3x4xf32>) outs(%empty_x : tensor<2x3x4xf32>) {
  ^bb0(%xv: f32, %o: f32):
    %n = arith.negf %xv : f32
    linalg.yield %n : f32
  } -> tensor<2x3x4xf32>
  %flat = tensor.collapse_shape %neg [[0,1],[2]]
    : tensor<2x3x4xf32> into tensor<6x4xf32>
  %empty = tensor.empty() : tensor<6x5xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<6x5xf32>) -> tensor<6x5xf32>
  %out = linalg.generic {
    indexing_maps = [affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>,
                    affine_map<(i,j,k)->(i,j)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%flat, %w : tensor<6x4xf32>, tensor<4x5xf32>) outs(%init : tensor<6x5xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %c, %m : f32
    linalg.yield %s : f32
  } -> tensor<6x5xf32>
  %expanded = tensor.expand_shape %out [[0,1],[2]] output_shape [2,3,5]
    : tensor<6x5xf32> into tensor<2x3x5xf32>
  return %expanded : tensor<2x3x5xf32>
}
"""

REDUCTION_COLLAPSE = COLLAPSE.replace("tensor<2x3x4xf32>", "tensor<6x2x2xf32>").replace(
    "tensor.collapse_shape %neg [[0,1],[2]]",
    "tensor.collapse_shape %neg [[0],[1,2]]",
)

EXPAND = """
func.func @indices(%x: tensor<6x4xf32>, %bias: tensor<6xf32>) -> tensor<4x2x3xf32> {
  %ten = arith.constant 10.0 : f32
  %empty = tensor.empty() : tensor<4x6xf32>
  %out:2 = linalg.generic {
    indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i)>,
                    affine_map<(i,j)->()>, affine_map<(i,j)->(j,i)>,
                    affine_map<(i,j)->(j,i)>],
    iterator_types = ["parallel", "parallel"], test.marker}
    ins(%x, %bias, %ten : tensor<6x4xf32>, tensor<6xf32>, f32)
    outs(%empty, %empty : tensor<4x6xf32>, tensor<4x6xf32>) {
  ^bb0(%a: f32, %b: f32, %t: f32, %o0: f32, %o1: f32):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %ii = arith.index_cast %i : index to i64
    %ji = arith.index_cast %j : index to i64
    %if = arith.sitofp %ii : i64 to f32
    %jf = arith.sitofp %ji : i64 to f32
    %it = arith.mulf %if, %t : f32
    %ij = arith.addf %it, %jf : f32
    %ab = arith.addf %a, %b : f32
    %result = arith.addf %ab, %ij : f32
    %neg = arith.negf %a : f32
    linalg.yield %result, %neg : f32, f32
  } -> (tensor<4x6xf32>, tensor<4x6xf32>)
  %expanded = tensor.expand_shape %out#0 [[0],[1,2]] output_shape [4,2,3]
    : tensor<4x6xf32> into tensor<4x2x3xf32>
  return %expanded : tensor<4x2x3xf32>
}
"""


def schedule(*, cleanup=True):
    with schedule_boilerplate() as (module, sequence):
        target = transform_ext.fold_reshape_ops_by_expansion(sequence.bodyTarget)
        if cleanup:
            lh_transform.cleanup(target)
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


def apply(source, *, cleanup=True):
    module = ir.Module.parse(source)
    transforms = schedule(cleanup=cleanup)
    transforms.body.operations[0].apply(module.operation)
    assert module.operation.verify()
    return module


def test_structure():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = apply(COLLAPSE)
        preprocess, contraction = operations(module, "linalg.generic")
        assert contraction.inputs[0] == preprocess.results[0]
        assert list(contraction.inputs[0].type.shape) == [2, 3, 4]
        assert list(contraction.results[0].type.shape) == [2, 3, 5]
        assert len(contraction.iterator_types) == 4
        assert [
            op.operation.name for op in contraction.regions[0].blocks[0].operations
        ] == ["arith.mulf", "arith.addf", "linalg.yield"]
        assert not operations(module, "tensor.collapse_shape")
        assert all(
            op.src.owner.operation.name == "tensor.empty"
            for op in operations(module, "tensor.expand_shape")
        )

        shared = (
            EXPAND.replace(
                "  return %expanded",
                "  func.call @observe(%out#0, %out#1) : "
                "(tensor<4x6xf32>, tensor<4x6xf32>) -> ()\n  return %expanded",
            )
            + "\nfunc.func private @observe(tensor<4x6xf32>, tensor<4x6xf32>)"
        )
        module = apply(shared, cleanup=False)
        (generic,) = operations(module, "linalg.generic")
        assert "test.marker" in generic.attributes
        assert len(generic.results) == 2
        assert all(list(r.type.shape) == [4, 2, 3] for r in generic.results)
        maps = indexing_maps(generic)
        assert maps[0] == ir.AffineMap.get_identity(3)
        assert [e.position for e in maps[1].results] == [0, 1]
        assert not maps[2].results
        assert [e.position for e in maps[3].results] == [2, 0, 1]
        (call,) = operations(module, "func.call")
        for i, operand in enumerate(call.operands):
            assert operand.owner.operation.name == "tensor.collapse_shape"
            assert operand.owner.src == generic.results[i]
        (ret,) = [op for op in operations(module, "func.return")]
        assert ret.operands[0] == generic.results[0]
        before = str(module)
        transforms = schedule(cleanup=False)
        transforms.body.operations[0].apply(module.operation)
        assert str(module) == before, "reshape folding must reach a fixed point"

        module = apply(REDUCTION_COLLAPSE)
        _, contraction = operations(module, "linalg.generic")
        assert list(contraction.inputs[0].type.shape) == [2, 3, 2, 2]
        assert [str(it) for it in contraction.iterator_types].count(
            "#linalg.iterator_type<reduction>"
        ) == 2
    print(
        "Reshape folding preserves contraction bodies, maps, attributes and shared results"
    )


# CHECK: Reshape folding preserves contraction bodies, maps, attributes and shared results


def test_unsupported():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        # Dynamic operands and non-projected maps must not be partially rewritten.
        dynamic = EXPAND.replace("tensor<6x4xf32>", "tensor<?x4xf32>")
        nonprojected = EXPAND.replace(
            "affine_map<(i,j)->(i)>", "affine_map<(i,j)->(i floordiv 2)>"
        ).replace("tensor<6xf32>", "tensor<3xf32>")
        nested = EXPAND.replace(
            "    %neg = arith.negf %a : f32",
            """    %true = arith.constant true
    %neg = scf.if %true -> (f32) {
      scf.yield %a : f32
    } else {
      scf.yield %b : f32
    }""",
        )
        for source in (dynamic, nonprojected, nested):
            module = ir.Module.parse(source)
            before = str(module)
            transforms = schedule(cleanup=False)
            transforms.body.operations[0].apply(module.operation)
            assert module.operation.verify()
            assert str(module) == before
    print("Unsupported reshape candidates are unchanged")


# CHECK: Unsupported reshape candidates are unchanged


def test_numerics():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8
    w = np.arange(20, dtype=np.float32).reshape(4, 5) / 8
    bias = np.arange(6, dtype=np.float32) / 4
    indexed = (
        x.reshape(6, 4)
        + bias[:, None]
        + np.arange(6, dtype=np.float32)[:, None] * 10
        + np.arange(4, dtype=np.float32)[None, :]
    ).T.reshape(4, 2, 3)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for source, args, expected in (
            (COLLAPSE, [x, w], -x @ w),
            (REDUCTION_COLLAPSE, [x.reshape(6, 2, 2), w], -x @ w),
            (EXPAND, [x.reshape(6, 4), bias], indexed),
        ):
            for transformed in (False, True):
                runner, name = compile_payload(
                    source, schedule() if transformed else None
                )
                output = np.empty(expected.shape, dtype=np.float32)
                runner.execute(name, args + [output])
                np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    print("CPU results agree before and after expansion, including linalg.index")


# CHECK: CPU results agree before and after expansion, including linalg.index


def test_normalization_schedule():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(reshaped_source())
        with schedule_boilerplate() as (transforms, sequence):
            func = lh_transform.match_op(sequence.bodyTarget, "func.func")
            func = transform_ext.fold_reshape_ops_by_expansion(func)
            lh_transform.cleanup(func)
            contractions = transform_ext.filter_contraction_ops(
                lh_transform.match_op(func, "linalg.generic")
            )
            transform_ext.sink_normalization_past_contraction(contractions)
            lh_transform.cleanup(func)
            transform.yield_([])
        transforms.body.operations[0].apply(module.operation)
        assert module.operation.verify()
        generics = operations(module, "linalg.generic")
        assert all(
            m.is_projected_permutation for op in generics for m in indexing_maps(op)
        )
        (contraction,) = [
            op for op in generics if "reduction" in str(op.iterator_types)
        ]
        assert list(contraction.results[0].type.shape) == [2, 2, 3]
        assert [
            op.operation.name for op in contraction.regions[0].blocks[0].operations
        ] == ["arith.mulf", "arith.addf", "linalg.yield"]
        assert contraction.inputs[0].owner.operation.name == "linalg.generic"
        assert any(
            op.operation.name == "arith.select"
            for op in contraction.inputs[0].owner.regions[0].blocks[0].operations
        )
    print(
        "Reshape folding before normalization sinking keeps projected maps and a separate mask"
    )


# CHECK: Reshape folding before normalization sinking keeps projected maps and a separate mask


if __name__ == "__main__":
    test_structure()
    test_unsupported()
    test_numerics()
    test_normalization_schedule()

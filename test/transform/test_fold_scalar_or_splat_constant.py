# RUN: %PYTHON %s | FileCheck %s

"""Scalar and splat-constant generic inputs are inlined into the body."""

import numpy as np
from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
from lighthouse.utils.mlir import linalg_inputs
from test_attention_mask_numerics import compile_payload


PAYLOAD = """
#rc = affine_map<(i, j) -> (i, j)>
#r = affine_map<(i, j) -> (i)>
func.func @select(%x: tensor<4x8xf32>, %mask: tensor<4xi1>) -> tensor<4x8xf32> {
  %zeros = arith.constant dense<0.0> : tensor<4x16xf32>
  %zero_tile = tensor.extract_slice %zeros[0, 8] [4, 8] [1, 1]
      : tensor<4x16xf32> to tensor<4x8xf32>
  %scale = arith.constant 2.0 : f32
  %ramp = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
      indexing_maps = [#r, #rc, #rc, affine_map<(i, j) -> ()>, #r, #rc],
      iterator_types = ["parallel", "parallel"]}
      ins(%mask, %zero_tile, %x, %scale, %ramp
          : tensor<4xi1>, tensor<4x8xf32>, tensor<4x8xf32>, f32, tensor<4xf32>)
      outs(%empty : tensor<4x8xf32>) attrs = {keep_me} {
  ^bb0(%m: i1, %z: f32, %xv: f32, %s: f32, %rv: f32, %o: f32):
    %scaled = arith.mulf %xv, %s : f32
    %shifted = arith.addf %scaled, %rv : f32
    %sel = arith.select %m, %z, %shifted : f32
    linalg.yield %sel : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
"""

# Non-splat constants and runtime values must stay operands.
UNCHANGED = """
#rc = affine_map<(i, j) -> (i, j)>
#r = affine_map<(i, j) -> (i)>
func.func @add(%x: tensor<4x8xf32>, %s: f32) -> tensor<4x8xf32> {
  %ramp = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %out = linalg.generic {
      indexing_maps = [#rc, #r, affine_map<(i, j) -> ()>, #rc],
      iterator_types = ["parallel", "parallel"]}
      ins(%x, %ramp, %s : tensor<4x8xf32>, tensor<4xf32>, f32)
      outs(%empty : tensor<4x8xf32>) {
  ^bb0(%xv: f32, %rv: f32, %sv: f32, %o: f32):
    %a = arith.addf %xv, %rv : f32
    %b = arith.mulf %a, %sv : f32
    linalg.yield %b : f32
  } -> tensor<4x8xf32>
  return %out : tensor<4x8xf32>
}
"""


def schedule():
    with schedule_boilerplate() as (sched, seq):
        func = lh_transform.match_op(seq.bodyTarget, "func.func")
        func = transform_ext.fold_scalar_or_splat_constant(func)
        lh_transform.cleanup(func)
        transform.yield_([])
    return sched


def operations(module, name):
    found = []

    def visit(op):
        if op.name == name:
            found.append(op.opview)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visit)
    return found


def test_structure():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(PAYLOAD)
        sched = schedule()
        sched.body.operations[0].apply(module.operation)
        assert module.operation.verify()
        (generic,) = operations(module, "linalg.generic")
        # The non-splat constant stays an operand; splat and scalar are inlined.
        assert [str(v.type) for v in linalg_inputs(generic)] == [
            "tensor<4xi1>",
            "tensor<4x8xf32>",
            "tensor<4xf32>",
        ]
        assert "keep_me" in generic.operation.attributes
        print(module)


# CHECK-DAG:  %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
# CHECK-DAG:  %[[TWO:.+]] = arith.constant 2.000000e+00 : f32
# CHECK-NOT:  tensor<4x16xf32>
# CHECK:      linalg.generic
# CHECK-SAME:   ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<4xi1>, tensor<4x8xf32>, tensor<4xf32>)
# CHECK-SAME:   attrs = {keep_me}
# CHECK:      ^bb0(%{{.+}}: i1, %{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
# CHECK:        arith.mulf %{{.+}}, %[[TWO]] : f32
# CHECK:        arith.select %{{.+}}, %[[ZERO]], %{{.+}} : f32


def test_unchanged_without_constants():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(UNCHANGED)
        before = str(module)
        sched = schedule()
        sched.body.operations[0].apply(module.operation)
        assert str(module) == before
    print("Generics without scalar or splat constant inputs are unchanged")


# CHECK: Generics without scalar or splat constant inputs are unchanged


def test_numerics():
    x = np.arange(32, dtype=np.float32).reshape(4, 8) / 8 - 1
    mask = np.array([True, False, True, False])
    expected = np.where(
        mask[:, None], 0.0, 2 * x + np.array([1, 2, 3, 4], np.float32)[:, None]
    ).astype(np.float32)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        for sched in (None, schedule()):
            runner, name = compile_payload(PAYLOAD, sched)
            output = np.empty_like(expected)
            runner.execute(name, [x, mask, output])
            np.testing.assert_array_equal(output, expected)
    print("Folding constants preserves CPU results")


# CHECK: Folding constants preserves CPU results


if __name__ == "__main__":
    test_structure()
    test_unchanged_without_constants()
    test_numerics()

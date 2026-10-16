# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def run(name, payload):
    print(f"Test: {name}", flush=True)

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        module = ir.Module.parse(payload)

        with schedule_boilerplate() as (sched, named_seq):
            ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
            transform_ext.remove_asserts(ops)
            lh_transform.cleanup(named_seq.bodyTarget)
            transform.yield_()

        sched.body.operations[0].apply(module.operation)
        print(module)


ASSERT_IN_LINALG = r"""
module {
  func.func @main(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<16xf32>

    %result = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
      ins(%arg0 : tensor<16xf32>)
      outs(%empty : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %nonzero = arith.cmpf une, %in, %c0 : f32
      cf.assert %nonzero, "division by zero"

      %div = arith.divf %c1, %in : f32
      linalg.yield %div : f32
    } -> tensor<16xf32>

    return %result : tensor<16xf32>
  }
}
"""


ASSERT_IN_FUNC = r"""
module {
  func.func @main(%arg0: i1, %arg1: tensor<16xf32>) -> tensor<16xf32> {
    cf.assert %arg0, "function assert"
    %c2 = arith.constant 2.0 : f32

    %empty = tensor.empty() : tensor<16xf32>
    %result = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>,
                        affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]}
      ins(%arg1 : tensor<16xf32>)
      outs(%empty : tensor<16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %res = arith.mulf %in, %c2 : f32
      linalg.yield %res : f32
    } -> tensor<16xf32>

    return %result : tensor<16xf32>
  }
}
"""


# CHECK-LABEL: Test: remove_assert_inside_linalg
# CHECK: linalg.generic
# CHECK-NOT: arith.cmpf
# CHECK: arith.divf
# CHECK-NOT: cf.assert
run("remove_assert_inside_linalg", ASSERT_IN_LINALG)


# CHECK-LABEL: Test: preserve_function_assert
# CHECK: cf.assert
# CHECK: linalg.generic
run("preserve_function_assert", ASSERT_IN_FUNC)

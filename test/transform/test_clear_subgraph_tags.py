# RUN: %PYTHON %s | FileCheck %s

"""Test that subgraph tagging attributes can be removed from a payload."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


PAYLOAD = """
module {
  func.func @payload(%arg: tensor<16x32xf32>, %out: tensor<16x32xf32>)
                     -> tensor<16x32xf32> {
    %result = linalg.generic {
        indexing_maps=[affine_map<(m,n)->(m,n)>,
                       affine_map<(m,n)->(m,n)>],
        iterator_types=["parallel","parallel"]}
      ins(%arg : tensor<16x32xf32>) outs(%out : tensor<16x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %neg = arith.negf %x : f32
      linalg.yield %neg : f32
    } -> tensor<16x32xf32>
    return %result : tensor<16x32xf32>
  }
}
"""


def test():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        with schedule_boilerplate() as (schedule, named_sequence):
            func = lh_transform.match_op(named_sequence.bodyTarget, "func.func")
            transform_ext.tag_subgraphs(func, kinds=["elementwise"])
            transform_ext.clear_subgraph_tags(func)
            transform.yield_()
        schedule.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK-NOT: transform_ext.subgraph_
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.subgraph_


test()

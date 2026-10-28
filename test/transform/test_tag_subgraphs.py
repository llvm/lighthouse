# RUN: %PYTHON %s | FileCheck %s

"""Test priority-ordered subgraph tagging."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


PAYLOAD = """
module {
  func.func @payload(%a: tensor<128x64xf32>, %b: tensor<64x32xf32>,
                     %out: tensor<128x32xf32>) -> tensor<128x32xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x32xf32>
    %init = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<128x32xf32>) -> tensor<128x32xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x32xf32>)
        outs(%init : tensor<128x32xf32>) -> tensor<128x32xf32>
    %result = linalg.generic {
        indexing_maps=[affine_map<(m,n)->(m,n)>,
                       affine_map<(m,n)->(m,n)>],
        iterator_types=["parallel","parallel"]}
      ins(%mm : tensor<128x32xf32>)
      outs(%out : tensor<128x32xf32>) {
    ^bb0(%x: f32, %y: f32):
      %relu = arith.maximumf %x, %zero : f32
      linalg.yield %relu : f32
    } -> tensor<128x32xf32>
    return %result : tensor<128x32xf32>
  }
}
"""


def test():
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        with schedule_boilerplate() as (schedule, named_sequence):
            func = lh_transform.match_op(named_sequence.bodyTarget, "func.func")
            transform_ext.tag_subgraphs(func, kinds=["gemm", "elementwise"])
            transform.yield_()
        schedule.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK: linalg.fill {transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}
# CHECK: linalg.matmul {transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}
# CHECK: linalg.generic
# CHECK-SAME: attrs = {transform_ext.subgraph_anchor, transform_ext.subgraph_id = 1 : i64, transform_ext.subgraph_kind = "gemm"}


test()

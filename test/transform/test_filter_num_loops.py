# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def apply_filter(payload: str, num_loops: int, name: str):
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload)
        with schedule_boilerplate() as (sched, named_seq):
            candidates = lh_transform.match_op(
                named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
            )
            filtered = transform_ext.filter_num_loops(
                candidates,
                num_loops,
            )
            transform.print_(target=filtered, name=name)
            transform.yield_()
        sched.body.operations[0].apply(module.operation)


PAYLOAD = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(
      %a: tensor<64xf32>,
      %b: tensor<8x8xf32>,
      %x: tensor<2x4x8xf32>,
      %y: tensor<2x8x4xf32>) -> (tensor<64xf32>, tensor<8x8xf32>, tensor<2x4x4xf32>) {
    %e1 = tensor.empty() : tensor<64xf32>
    %add = linalg.add ins(%a, %a : tensor<64xf32>, tensor<64xf32>)
        outs(%e1 : tensor<64xf32>) -> tensor<64xf32>

    %e2 = tensor.empty() : tensor<8x8xf32>
    %gen = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%b : tensor<8x8xf32>)
        outs(%e2 : tensor<8x8xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<8x8xf32>

    %e3 = tensor.empty() : tensor<2x4x4xf32>
    %bm = linalg.batch_matmul ins(%x, %y : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
        outs(%e3 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    return %add, %gen, %bm : tensor<64xf32>, tensor<8x8xf32>, tensor<2x4x4xf32>
  }
}
"""


# CHECK-LABEL: IR printer: AT_LEAST_2
# CHECK: linalg.generic
# CHECK: linalg.batch_matmul
# CHECK-NOT: linalg.add
apply_filter(PAYLOAD, num_loops=2, name="AT_LEAST_2")

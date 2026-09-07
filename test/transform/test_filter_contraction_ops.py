# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate


def apply_filter(payload: str, name: str):
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        module = ir.Module.parse(payload)
        with schedule_boilerplate() as (sched, named_seq):
            candidates = lh_transform.match_op(
                named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
            )
            filtered = transform_ext.filter_contraction_ops(candidates)
            transform.print_(target=filtered, name=name)
            transform.yield_()
        sched.body.operations[0].apply(module.operation)


# Named matmul and batch_matmul mixed with elementwise ops.
NAMED = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(
      %a: tensor<8x8xf32>,
      %m0: tensor<4x8xf32>,
      %m1: tensor<8x4xf32>,
      %x: tensor<2x4x8xf32>,
      %y: tensor<2x8x4xf32>)
      -> (tensor<8x8xf32>, tensor<4x4xf32>, tensor<2x4x4xf32>) {
    %e0 = tensor.empty() : tensor<8x8xf32>
    %add = linalg.add ins(%a, %a : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%e0 : tensor<8x8xf32>) -> tensor<8x8xf32>

    %e1 = tensor.empty() : tensor<8x8xf32>
    %gen = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<8x8xf32>)
        outs(%e1 : tensor<8x8xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<8x8xf32>

    %e2 = tensor.empty() : tensor<4x4xf32>
    %mm = linalg.matmul ins(%m0, %m1 : tensor<4x8xf32>, tensor<8x4xf32>)
        outs(%e2 : tensor<4x4xf32>) -> tensor<4x4xf32>

    %e3 = tensor.empty() : tensor<2x4x4xf32>
    %bm = linalg.batch_matmul ins(%x, %y : tensor<2x4x8xf32>, tensor<2x8x4xf32>)
        outs(%e3 : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    return %gen, %mm, %bm : tensor<8x8xf32>, tensor<4x4xf32>, tensor<2x4x4xf32>
  }
}
"""

# CHECK-LABEL: IR printer: NAMED
# CHECK: linalg.matmul
# CHECK: linalg.batch_matmul
# CHECK-NOT: linalg.add
# CHECK-NOT: linalg.generic
apply_filter(NAMED, name="NAMED")


# A linalg.generic with elementwise (extf) ops fused into the matmul body,
# preceded by a linalg.fill that must not be matched.
FUSED = """
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @main(
      %a: tensor<1x1x128x1024xbf16>,
      %b: tensor<1x1x1024x512xbf16>) -> tensor<1x1x128x512xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<1x1x128x512xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%e : tensor<1x1x128x512xf32>) -> tensor<1x1x128x512xf32>
    %mm = linalg.generic {indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<1x1x128x1024xbf16>, tensor<1x1x1024x512xbf16>)
        outs(%fill : tensor<1x1x128x512xf32>) {
    ^bb0(%in: bf16, %in_18: bf16, %out: f32):
      %ea = arith.extf %in : bf16 to f32
      %eb = arith.extf %in_18 : bf16 to f32
      %mul = arith.mulf %ea, %eb : f32
      %acc = arith.addf %out, %mul : f32
      linalg.yield %acc : f32
    } -> tensor<1x1x128x512xf32>
    return %mm : tensor<1x1x128x512xf32>
  }
}
"""

# CHECK-LABEL: IR printer: FUSED
# CHECK: linalg.generic
# CHECK-NOT: linalg.fill
apply_filter(FUSED, name="FUSED")

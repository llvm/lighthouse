# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, build_schedule):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


PAYLOAD = """
module {
    func.func @main(%a: tensor<16x8xf32>, %b: tensor<8x16xf32>) -> tensor<16x16xf32> {
    %cst = arith.constant 0.0 : f32
        %e = tensor.empty() : tensor<16x16xf32>
        %f = linalg.fill ins(%cst : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
        %mm = linalg.matmul ins(%a, %b : tensor<16x8xf32>, tensor<8x16xf32>)
                outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
        return %mm : tensor<16x16xf32>
  }
}
"""


def build_schedule(op_name: str = "linalg.matmul"):
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, op_name)
        assign_tile_sizes(
            ops,
            strategy="register_unroll",
        )
        transform.yield_()
    return sched


# CHECK-LABEL: Test: register_unroll_strategy
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 1>
run("register_unroll_strategy", PAYLOAD, build_schedule)


# A non-contraction reduction takes both tiles from the generic fallback, which
# the purely elementwise cases never exercise. The target is pinned because the
# generic parallel tile follows the SIMD width.
GENERIC_REDUCE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x256xf32>, %o: tensor<64xf32>) -> tensor<64xf32> {
    %r = linalg.generic {indexing_maps = [#id, #out],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<64x256xf32>) outs(%o : tensor<64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
"""

# CHECK-LABEL: Test: register_unroll_generic_reduce
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 1>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_generic_reduce",
        GENERIC_REDUCE,
        lambda: build_schedule("linalg.generic"),
    )


# Same reduction on bf16: the generic parallel tile doubles with the narrower
# element type, while the reduction tile is width-independent.
GENERIC_REDUCE_BF16 = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @main(%a: tensor<64x256xbf16>, %o: tensor<64xbf16>) -> tensor<64xbf16> {
    %r = linalg.generic {indexing_maps = [#id, #out],
        iterator_types = ["parallel", "reduction"]}
        ins(%a : tensor<64x256xbf16>) outs(%o : tensor<64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %s = arith.addf %in, %out : bf16
      linalg.yield %s : bf16
    } -> tensor<64xbf16>
    return %r : tensor<64xbf16>
  }
}
"""

# CHECK-LABEL: Test: register_unroll_generic_reduce_bf16
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 1>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_generic_reduce_bf16",
        GENERIC_REDUCE_BF16,
        lambda: build_schedule("linalg.generic"),
    )

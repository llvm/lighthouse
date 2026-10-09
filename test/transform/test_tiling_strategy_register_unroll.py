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


# A row (inner) reduction keeps its reduced vector dim whole so it lowers to a
# single wide horizontal reduction; other dims are unrolled to 1.
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
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_generic_reduce",
        GENERIC_REDUCE,
        lambda: build_schedule("linalg.generic"),
    )


# A reduced dim shorter than a vector (e.g. a pooling window) is not kept whole
# as one horizontal reduction: the generic unroll tiling vectorizes the rows.
# CHECK-LABEL: Test: register_unroll_short_reduce
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 1>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_short_reduce",
        GENERIC_REDUCE.replace("64x256", "64x4"),
        lambda: build_schedule("linalg.generic"),
    )


# A column (outer) reduction unrolls to one vector register along its
# contiguous parallel dim.
COLUMN_REDUCE = """
#id = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%a: tensor<256x64xELEM>, %o: tensor<64xELEM>) -> tensor<64xELEM> {
    %r = linalg.generic {indexing_maps = [#id, #out],
        iterator_types = ["reduction", "parallel"]}
        ins(%a : tensor<256x64xELEM>) outs(%o : tensor<64xELEM>) {
    ^bb0(%in: ELEM, %out: ELEM):
      %s = arith.addf %in, %out : ELEM
      linalg.yield %s : ELEM
    } -> tensor<64xELEM>
    return %r : tensor<64xELEM>
  }
}
"""

# CHECK-LABEL: Test: register_unroll_column_reduce_avx512
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_column_reduce_avx512",
        COLUMN_REDUCE.replace("ELEM", "f32"),
        lambda: build_schedule("linalg.generic"),
    )

# CHECK-LABEL: Test: register_unroll_column_reduce_avx2
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 8>
with TargetInfo.override(features=["avx2"]):
    run(
        "register_unroll_column_reduce_avx2",
        COLUMN_REDUCE.replace("ELEM", "f32"),
        lambda: build_schedule("linalg.generic"),
    )


# bf16 is computed as f32, so it unrolls to the same number of lanes as f32.
# CHECK-LABEL: Test: register_unroll_column_reduce_bf16
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=["avx512f"]):
    run(
        "register_unroll_column_reduce_bf16",
        COLUMN_REDUCE.replace("ELEM", "bf16"),
        lambda: build_schedule("linalg.generic"),
    )

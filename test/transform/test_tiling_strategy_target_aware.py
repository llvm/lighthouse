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


F32_MATMUL = """
module {
  func.func @main(%a: tensor<128x64xf32>, %b: tensor<64x128xf32>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128x128xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xf32>, tensor<64x128xf32>)
        outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
  }
}
"""


BF16_MATMUL = """
module {
  func.func @main(%a: tensor<128x64xbf16>, %b: tensor<64x128xbf16>) -> tensor<128x128xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<128x128xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<128x128xf32>) -> tensor<128x128xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<128x64xbf16>, tensor<64x128xbf16>)
        outs(%f : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mm : tensor<128x128xf32>
  }
}
"""


# A non-contraction op, so register_parallel falls back to generic_parallel_tiles,
# whose inner tile is the target's SIMD lane count for the output element type.
ELTWISE = """
module {
    func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %sum = linalg.add ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                outs(%a : tensor<64x64xf32>) -> tensor<64x64xf32>
        return %sum : tensor<64x64xf32>
    }
}
"""


def build_register_parallel():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


def build_register_reduction():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        assign_tile_sizes(
            ops,
            strategy="register_reduction",
        )
        transform.yield_()
    return sched


def build_register_unroll():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.matmul")
        assign_tile_sizes(
            ops,
            strategy="register_unroll",
        )
        transform.yield_()
    return sched


# CHECK-LABEL: Test: f32_register_parallel_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32, 0>
run("f32_register_parallel_default", F32_MATMUL, lambda: build_register_parallel())


# CHECK-LABEL: Test: bf16_amx_register_parallel_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32, 0>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "bf16_amx_register_parallel_default",
        BF16_MATMUL,
        lambda: build_register_parallel(),
    )


# Without AMX, bf16 matmul falls back to the generic SIMD-lane tiling.
# CHECK-LABEL: Test: bf16_no_amx_register_parallel_generic_fallback
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 0>
with TargetInfo.override(features=[]):
    run(
        "bf16_no_amx_register_parallel_generic_fallback",
        BF16_MATMUL,
        lambda: build_register_parallel(),
    )


def build_register_parallel_eltwise():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.add")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


# 512-bit vectors (AVX-512): 32-bit lanes -> inner tile of 16.
# CHECK-LABEL: Test: eltwise_register_parallel_avx512
# CHECK: linalg.add
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512",
        ELTWISE,
        lambda: build_register_parallel_eltwise(),
    )


# 256-bit vectors (AVX2): 32-bit lanes -> inner tile of 8.
# CHECK-LABEL: Test: eltwise_register_parallel_avx2
# CHECK: linalg.add
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 8>
with TargetInfo.override(features=["avx2"]):
    run(
        "eltwise_register_parallel_avx2",
        ELTWISE,
        lambda: build_register_parallel_eltwise(),
    )


# 128-bit vectors (SSE): 32-bit lanes -> inner tile of 4.
# CHECK-LABEL: Test: eltwise_register_parallel_sse
# CHECK: linalg.add
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 4>
with TargetInfo.override(features=["sse4_1"]):
    run(
        "eltwise_register_parallel_sse",
        ELTWISE,
        lambda: build_register_parallel_eltwise(),
    )


# No recognized vector extension: falls back to the 512-bit assumption -> 16.
# CHECK-LABEL: Test: eltwise_register_parallel_no_features
# CHECK: linalg.add
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16>
with TargetInfo.override(features=[]):
    run(
        "eltwise_register_parallel_no_features",
        ELTWISE,
        lambda: build_register_parallel_eltwise(),
    )


# CHECK-LABEL: Test: f32_register_reduction_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 2>
run("f32_register_reduction_default", F32_MATMUL, lambda: build_register_reduction())


# CHECK-LABEL: Test: bf16_amx_register_reduction_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 32>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "bf16_amx_register_reduction_default",
        BF16_MATMUL,
        lambda: build_register_reduction(),
    )


# Without AMX, bf16 matmul is not an all-f32 contraction either, so it falls back
# to the generic reduction tile.
# CHECK-LABEL: Test: bf16_no_amx_register_reduction_generic_fallback
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 1>
with TargetInfo.override(features=[]):
    run(
        "bf16_no_amx_register_reduction_generic_fallback",
        BF16_MATMUL,
        lambda: build_register_reduction(),
    )


# An AMX-capable target must not change f32 GEMM reduction tiling.
# CHECK-LABEL: Test: f32_register_reduction_under_amx_target
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 2>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "f32_register_reduction_under_amx_target",
        F32_MATMUL,
        lambda: build_register_reduction(),
    )

# CHECK-LABEL: Test: f32_register_unroll_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 1>
run("f32_register_unroll_default", F32_MATMUL, lambda: build_register_unroll())


# CHECK-LABEL: Test: bf16_amx_register_unroll_default
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 16, 32>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "bf16_amx_register_unroll_default",
        BF16_MATMUL,
        lambda: build_register_unroll(),
    )


# Without AMX, bf16 is neither an AMX nor an all-f32 contraction, so both the
# parallel and reduction tiles come from the generic fallback.
# CHECK-LABEL: Test: bf16_no_amx_register_unroll_generic_fallback
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 1>
with TargetInfo.override(features=[]):
    run(
        "bf16_no_amx_register_unroll_generic_fallback",
        BF16_MATMUL,
        lambda: build_register_unroll(),
    )


# An AMX-capable target must not change f32 GEMM unroll tiling.
# CHECK-LABEL: Test: f32_register_unroll_under_amx_target
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 1>
with TargetInfo.override(features=["amx_tile"]):
    run(
        "f32_register_unroll_under_amx_target",
        F32_MATMUL,
        lambda: build_register_unroll(),
    )

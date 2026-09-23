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
  func.func @main(%a: tensor<4x64x64xf32>, %b: tensor<4x64x64xf32>) -> tensor<4x64x64xf32> {
    %cst = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<4x64x64xf32>
    %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
    %mm = linalg.batch_matmul ins(%a, %b : tensor<4x64x64xf32>, tensor<4x64x64xf32>)
        outs(%f : tensor<4x64x64xf32>) -> tensor<4x64x64xf32>
    return %mm : tensor<4x64x64xf32>
  }
}
"""


def build_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.batch_matmul")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


def build_eltwise_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.elementwise")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


ELTWISE_1D = """
module {
  func.func @main(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<128xf32>, tensor<128xf32>)
            outs(%a : tensor<128xf32>) -> tensor<128xf32>
    return %sum : tensor<128xf32>
  }
}
"""

ELTWISE_2D = """
module {
  func.func @main(%a: tensor<64x32xf32>, %b: tensor<64x32xf32>) -> tensor<64x32xf32> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xf32>, tensor<64x32xf32>)
            outs(%a : tensor<64x32xf32>) -> tensor<64x32xf32>
    return %sum : tensor<64x32xf32>
  }
}
"""

ELTWISE_3D = """
module {
  func.func @main(%a: tensor<8x16x32xf32>, %b: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<8x16x32xf32>, tensor<8x16x32xf32>)
            outs(%a : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    return %sum : tensor<8x16x32xf32>
  }
}
"""

ELTWISE_4D = """
module {
  func.func @main(%a: tensor<4x8x16x32xf32>, %b: tensor<4x8x16x32xf32>) -> tensor<4x8x16x32xf32> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<4x8x16x32xf32>, tensor<4x8x16x32xf32>)
            outs(%a : tensor<4x8x16x32xf32>) -> tensor<4x8x16x32xf32>
    return %sum : tensor<4x8x16x32xf32>
  }
}
"""

ELTWISE_AVX512_F16_2D = """
module {
  func.func @main(%a: tensor<64x32xf16>, %b: tensor<64x32xf16>) -> tensor<64x32xf16> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xf16>, tensor<64x32xf16>)
            outs(%a : tensor<64x32xf16>) -> tensor<64x32xf16>
    return %sum : tensor<64x32xf16>
  }
}
"""

ELTWISE_AVX512_BF16_2D = """
module {
  func.func @main(%a: tensor<64x32xbf16>, %b: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xbf16>, tensor<64x32xbf16>)
            outs(%a : tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %sum : tensor<64x32xbf16>
  }
}
"""

ELTWISE_AVX512_F64_2D = """
module {
  func.func @main(%a: tensor<64x32xf64>, %b: tensor<64x32xf64>) -> tensor<64x32xf64> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xf64>, tensor<64x32xf64>)
            outs(%a : tensor<64x32xf64>) -> tensor<64x32xf64>
    return %sum : tensor<64x32xf64>
  }
}
"""

ELTWISE_AVX512_F32_TALL_SKINNY_1024X4 = """
module {
  func.func @main(%a: tensor<1024x4xf32>, %b: tensor<1024x4xf32>) -> tensor<1024x4xf32> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<1024x4xf32>, tensor<1024x4xf32>)
            outs(%a : tensor<1024x4xf32>) -> tensor<1024x4xf32>
    return %sum : tensor<1024x4xf32>
  }
}
"""

ELTWISE_AVX512_I8_2D = """
module {
  func.func @main(%a: tensor<64x32xi8>, %b: tensor<64x32xi8>) -> tensor<64x32xi8> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xi8>, tensor<64x32xi8>)
            outs(%a : tensor<64x32xi8>) -> tensor<64x32xi8>
    return %sum : tensor<64x32xi8>
  }
}
"""

ELTWISE_AVX512_I16_2D = """
module {
  func.func @main(%a: tensor<64x32xi16>, %b: tensor<64x32xi16>) -> tensor<64x32xi16> {
    %sum = linalg.elementwise <add>
            ins(%a, %b : tensor<64x32xi16>, tensor<64x32xi16>)
            outs(%a : tensor<64x32xi16>) -> tensor<64x32xi16>
    return %sum : tensor<64x32xi16>
  }
}
"""


# CHECK-LABEL: Test: register_parallel_strategy
# CHECK: linalg.batch_matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 8, 32, 0>
run("register_parallel_strategy", PAYLOAD, build_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx2_1d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 128>
with TargetInfo.override(features=["avx2"]):
    run("eltwise_register_parallel_avx2_1d", ELTWISE_1D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx2_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 4, 32>
with TargetInfo.override(features=["avx2"]):
    run("eltwise_register_parallel_avx2_2d", ELTWISE_2D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx2_3d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 4, 32>
with TargetInfo.override(features=["avx2"]):
    run("eltwise_register_parallel_avx2_3d", ELTWISE_3D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx2_4d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 4, 32>
with TargetInfo.override(features=["avx2"]):
    run("eltwise_register_parallel_avx2_4d", ELTWISE_4D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_sse_1d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 64>
with TargetInfo.override(features=["sse4_1"]):
    run("eltwise_register_parallel_sse_1d", ELTWISE_1D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_sse_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 2, 32>
with TargetInfo.override(features=["sse4_1"]):
    run("eltwise_register_parallel_sse_2d", ELTWISE_2D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_sse_3d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 2, 32>
with TargetInfo.override(features=["sse4_1"]):
    run("eltwise_register_parallel_sse_3d", ELTWISE_3D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_sse_4d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 2, 32>
with TargetInfo.override(features=["sse4_1"]):
    run("eltwise_register_parallel_sse_4d", ELTWISE_4D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_f16_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 32>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_f16_2d",
        ELTWISE_AVX512_F16_2D,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_bf16_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 32>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_bf16_2d",
        ELTWISE_AVX512_BF16_2D,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_f64_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 8, 32>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_f64_2d",
        ELTWISE_AVX512_F64_2D,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_f32_tall_skinny_1024x4
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 128, 4>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_f32_tall_skinny_1024x4",
        ELTWISE_AVX512_F32_TALL_SKINNY_1024X4,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_i8_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 64, 32>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_i8_2d",
        ELTWISE_AVX512_I8_2D,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_i16_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
with TargetInfo.override(features=["avx512f"]):
    run(
        "eltwise_register_parallel_avx512_i16_2d",
        ELTWISE_AVX512_I16_2D,
        build_eltwise_schedule,
    )

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_1d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 128>
with TargetInfo.override(features=["avx512f"]):
    run("eltwise_register_parallel_avx512_1d", ELTWISE_1D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 32>
with TargetInfo.override(features=["avx512f"]):
    run("eltwise_register_parallel_avx512_2d", ELTWISE_2D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_3d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 16, 32>
with TargetInfo.override(features=["avx512f"]):
    run("eltwise_register_parallel_avx512_3d", ELTWISE_3D, build_eltwise_schedule)

# CHECK-LABEL: Test: eltwise_register_parallel_avx512_4d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1, 16, 32>
with TargetInfo.override(features=["avx512f"]):
    run("eltwise_register_parallel_avx512_4d", ELTWISE_4D, build_eltwise_schedule)

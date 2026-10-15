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


def build_reduction_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
        assign_tile_sizes(
            ops,
            strategy="register_parallel",
        )
        transform.yield_()
    return sched


REDUCE_TEMPLATE = """
#in = affine_map<IN_MAP>
#out = affine_map<OUT_MAP>
module {
  func.func @main(%a: tensor<IN_TYPE>, %o: tensor<OUT_TYPE>) -> tensor<OUT_TYPE> {
    %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = [ITERS]}
        ins(%a : tensor<IN_TYPE>) outs(%o : tensor<OUT_TYPE>) {
    ^bb0(%in: f32, %out: f32):
      %s = arith.addf %in, %out : f32
      linalg.yield %s : f32
    } -> tensor<OUT_TYPE>
    return %r : tensor<OUT_TYPE>
  }
}
"""


def reduce_payload(in_map, out_map, iters, in_type, out_type):
    return (
        REDUCE_TEMPLATE.replace("IN_MAP", in_map)
        .replace("OUT_MAP", out_map)
        .replace("ITERS", iters)
        .replace("IN_TYPE", in_type)
        .replace("OUT_TYPE", out_type)
    )


ROW_REDUCE = reduce_payload(
    "(d0, d1) -> (d0, d1)",
    "(d0, d1) -> (d0)",
    '"parallel", "reduction"',
    "64x4096xf32",
    "64xf32",
)

PARTIAL_REDUCE = reduce_payload(
    "(d0, d1, d2) -> (d0, d1, d2)",
    "(d0, d1, d2) -> (d0, d2)",
    '"parallel", "reduction", "parallel"',
    "64x32x128xf32",
    "64x128xf32",
)


def column_reduce(shape):
    n, c, h, w = shape
    return reduce_payload(
        "(d0, d1, d2, d3) -> (d0, d1, d2, d3)",
        "(d0, d1, d2, d3) -> (d0, d2, d3)",
        '"parallel", "reduction", "parallel", "parallel"',
        f"{n}x{c}x{h}x{w}xf32",
        f"{n}x{h}x{w}xf32",
    )


# A long row reduction gets all accumulator chains from its reduced vector dim,
# so a single row is processed at a time.
# CHECK-LABEL: Test: reduction_register_parallel_row
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run("reduction_register_parallel_row", ROW_REDUCE, build_reduction_schedule)


# A reduced dim shorter than a vector (e.g. a pooling window) has no lanes to
# fill: the op is left to its neighbours' tiles.
# CHECK-LABEL: Test: reduction_register_parallel_short_row
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "reduction_register_parallel_short_row",
        reduce_payload(
            "(d0, d1) -> (d0, d1)",
            "(d0, d1) -> (d0)",
            '"parallel", "reduction"',
            "64x4xf32",
            "64xf32",
        ),
        build_reduction_schedule,
    )

# A split partial reduction keeps lanes x chains independent accumulators.
# CHECK-LABEL: Test: reduction_register_parallel_partial
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 128>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "reduction_register_parallel_partial",
        PARTIAL_REDUCE,
        build_reduction_schedule,
    )

# CHECK-LABEL: Test: reduction_register_parallel_column_avx2
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 1, 64>
with TargetInfo.override(arch="x86_64", features=["avx2"]):
    run(
        "reduction_register_parallel_column_avx2",
        column_reduce((4, 16, 64, 256)),
        build_reduction_schedule,
    )

# A narrow contiguous dim provides only 2 chains; the rest are spread over the
# next outer parallel dim.
# CHECK-LABEL: Test: reduction_register_parallel_column_narrow
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 0, 4, 32>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "reduction_register_parallel_column_narrow",
        column_reduce((2, 16, 8, 32)),
        build_reduction_schedule,
    )

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


# A contraction with two reduction dims (k0, k1): only the innermost one gets the
# strategy tile, the outer reduction dim is unit-tiled.
MULTI_REDUCTION = """
#mA = affine_map<(m, n, k0, k1) -> (m, k0, k1)>
#mB = affine_map<(m, n, k0, k1) -> (k0, k1, n)>
#mC = affine_map<(m, n, k0, k1) -> (m, n)>
module {
  func.func @main(%a: tensor<64x32x16xf32>, %b: tensor<32x16x128xf32>,
        %o: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %c = linalg.contract indexing_maps = [#mA, #mB, #mC]
        ins(%a, %b : tensor<64x32x16xf32>, tensor<32x16x128xf32>)
        outs(%o : tensor<64x128xf32>) -> tensor<64x128xf32>
    return %c : tensor<64x128xf32>
  }
}
"""


# A non-contraction row reduction: tiled by the reduction register heuristic.
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


# A fully parallel op: the strategy has nothing to tile.
ELTWISE = """
module {
  func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %s = linalg.elementwise <add>
        ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%a : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %s : tensor<64x64xf32>
  }
}
"""


def build_schedule(op_name: str = "linalg.batch_matmul"):
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, op_name)
        assign_tile_sizes(
            ops,
            strategy="register_reduction",
        )
        transform.yield_()
    return sched


# Parallel dims (batch, M, N) stay untiled; only the K dim gets the f32 tile.
# CHECK-LABEL: Test: register_reduction_strategy
# CHECK: linalg.batch_matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 0, 2>
run("register_reduction_strategy", PAYLOAD, build_schedule)


# CHECK-LABEL: Test: register_reduction_multiple_reduction_dims
# CHECK: linalg.contract
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 0, 1, 2>
run(
    "register_reduction_multiple_reduction_dims",
    MULTI_REDUCTION,
    lambda: build_schedule("linalg.contract"),
)


# A row reduction tiles its reduced vector dim by lanes x accumulator chains.
# CHECK-LABEL: Test: register_reduction_generic_reduce
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 128>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "register_reduction_generic_reduce",
        GENERIC_REDUCE,
        lambda: build_schedule("linalg.generic"),
    )

# CHECK-LABEL: Test: register_reduction_generic_reduce_avx2
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 64>
with TargetInfo.override(arch="x86_64", features=["avx2"]):
    run(
        "register_reduction_generic_reduce_avx2",
        GENERIC_REDUCE,
        lambda: build_schedule("linalg.generic"),
    )


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


# RMSNorm-like channel reduction: the contiguous dim is parallel, so only the
# reduced dim is unrolled by a small factor.
# CHECK-LABEL: Test: register_reduction_column
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 2, 0, 0>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "register_reduction_column",
        reduce_payload(
            "(d0, d1, d2, d3) -> (d0, d1, d2, d3)",
            "(d0, d1, d2, d3) -> (d0, d2, d3)",
            '"parallel", "reduction", "parallel", "parallel"',
            "4x16x64x256xf32",
            "4x64x256xf32",
        ),
        lambda: build_schedule("linalg.generic"),
    )


# LayerNorm-like all-reduce: innermost reduced dim gets the vector tile, the
# outer reduced dims are unit-tiled.
# CHECK-LABEL: Test: register_reduction_all_reduce
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 1, 1, 128>
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "register_reduction_all_reduce",
        reduce_payload(
            "(d0, d1, d2, d3) -> (d0, d1, d2, d3)",
            "(d0, d1, d2, d3) -> (d0)",
            '"parallel", "reduction", "reduction", "reduction"',
            "16x64x256x256xf32",
            "16xf32",
        ),
        lambda: build_schedule("linalg.generic"),
    )


# A short row reduction (e.g. the final one after a split) fits one horizontal
# reduction: nothing to tile, so the op stays unannotated.
# CHECK-LABEL: Test: register_reduction_short_row
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "register_reduction_short_row",
        reduce_payload(
            "(d0, d1) -> (d0, d1)",
            "(d0, d1) -> (d0)",
            '"parallel", "reduction"',
            "64x128xf32",
            "64xf32",
        ),
        lambda: build_schedule("linalg.generic"),
    )


# Reduced dims shorter than a vector (e.g. a 4x4 pooling window) have no lanes
# to fill: unit-tiling the outer window dim would unroll the op per element.
# CHECK-LABEL: Test: register_reduction_short_window
# CHECK: linalg.generic
# CHECK-NOT: transform_ext.tile_sizes
# CHECK: return
with TargetInfo.override(arch="x86_64", features=["avx512f"]):
    run(
        "register_reduction_short_window",
        reduce_payload(
            "(d0, d1, d2) -> (d0, d1, d2)",
            "(d0, d1, d2) -> (d0)",
            '"parallel", "reduction", "reduction"',
            "64x4x4xf32",
            "64xf32",
        ),
        lambda: build_schedule("linalg.generic"),
    )


# No reduction dim: the op must be left unannotated.
# CHECK-LABEL: Test: register_reduction_no_reduction_dims
# CHECK: linalg.elementwise
# CHECK-NOT: transform_ext.tile_sizes
run(
    "register_reduction_no_reduction_dims",
    ELTWISE,
    lambda: build_schedule("linalg.elementwise"),
)

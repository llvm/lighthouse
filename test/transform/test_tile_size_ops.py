# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.dialects.transform.transform_ext import get_leading_unit_tile_sizes
from lighthouse.dialects.transform.transform_ext import get_tile_sizes
from lighthouse.schedule.builders import schedule_boilerplate


def run(name, payload_str, build_schedule):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load(reload=True)
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


MATMUL_PAYLOAD = """
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


ANNOTATED_ELTWISE_PAYLOAD = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %e = tensor.empty() : tensor<16x32xf32>
    %g = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel"],
        transform_ext.tile_sizes = array<i64: 4, 8>}
        ins(%a : tensor<16x32xf32>)
        outs(%e : tensor<16x32xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<16x32xf32>
    return %g : tensor<16x32xf32>
  }
}
"""


SMALL_DIM_ELTWISE_PAYLOAD = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
    func.func @main(%a: tensor<16x64xf32>) -> tensor<16x64xf32> {
        %e = tensor.empty() : tensor<16x64xf32>
        %g = linalg.generic {indexing_maps = [#id, #id],
                iterator_types = ["parallel", "parallel"]}
                ins(%a : tensor<16x64xf32>)
                outs(%e : tensor<16x64xf32>) {
        ^bb0(%i: f32, %o: f32):
            linalg.yield %i : f32
        } -> tensor<16x64xf32>
        return %g : tensor<16x64xf32>
    }
}
"""


ADD_1D_PAYLOAD = """
module {
    func.func @main(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
        %e = tensor.empty() : tensor<128xf32>
        %r = linalg.add ins(%a, %b : tensor<128xf32>, tensor<128xf32>)
                outs(%e : tensor<128xf32>) -> tensor<128xf32>
        return %r : tensor<128xf32>
    }
}
"""


BATCH_MATMUL_PAYLOAD = """
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


def build_assign_schedule(op_name: str):
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, op_name)
        assign_tile_sizes(ops)
        transform.yield_()
    return sched


def build_get_tile_sizes_schedule(op_name: str):
    with schedule_boilerplate() as (sched, named_seq):
        generic = lh_transform.match_op(named_seq.bodyTarget, op_name)
        sizes = get_tile_sizes(generic)
        structured.TileUsingForallOp(generic, num_threads=[], tile_sizes=sizes)
        transform.yield_()
    return sched


def build_get_leading_unit_tile_sizes_schedule(op_name: str):
    with schedule_boilerplate() as (sched, named_seq):
        matmuls = lh_transform.match_op(named_seq.bodyTarget, op_name)
        sizes = get_leading_unit_tile_sizes(matmuls)
        structured.TileUsingForallOp(matmuls, num_threads=[], tile_sizes=sizes)
        transform.yield_()
    return sched


# CHECK-LABEL: Test: assign_tile_sizes
# CHECK: linalg.matmul {transform_ext.tile_sizes = array<i64: 32, 32, 0>}
run(
    "assign_tile_sizes",
    MATMUL_PAYLOAD,
    lambda: build_assign_schedule("linalg.matmul"),
)


# CHECK-LABEL: Test: get_tile_sizes
# CHECK: scf.forall
# CHECK-SAME: in (4, 4)
run(
    "get_tile_sizes",
    ANNOTATED_ELTWISE_PAYLOAD,
    lambda: build_get_tile_sizes_schedule("linalg.generic"),
)


# CHECK-LABEL: Test: get_leading_unit_tile_sizes
# CHECK: scf.forall
# CHECK-SAME: in (16)
run(
    "get_leading_unit_tile_sizes",
    SMALL_DIM_ELTWISE_PAYLOAD,
    lambda: build_get_leading_unit_tile_sizes_schedule("linalg.generic"),
)


# First static dim (16) is below default tile size (32), so _disable_small_tiles
# must disable tiling there while keeping the larger dim tiled.
# CHECK-LABEL: Test: disable_small_tiles
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 32>
run(
    "disable_small_tiles",
    SMALL_DIM_ELTWISE_PAYLOAD,
    lambda: build_assign_schedule("linalg.generic"),
)


# 1D op with only 1D operands: compute_tile_sizes should tile its only parallel
# dim with the default tile size.
# CHECK-LABEL: Test: compute_tile_sizes_1d_op
# CHECK: linalg.add {transform_ext.tile_sizes = array<i64: 32>}
run(
    "compute_tile_sizes_1d_op",
    ADD_1D_PAYLOAD,
    lambda: build_assign_schedule("linalg.add"),
)


# 3D operand op (batch_matmul): compute_tile_sizes should tile batch by 1, M/N by
# full tile size, and keep K untiled.
# CHECK-LABEL: Test: compute_tile_sizes_3d_operands
# CHECK: linalg.batch_matmul {transform_ext.tile_sizes = array<i64: 1, 32, 32, 0>}
run(
    "compute_tile_sizes_3d_operands",
    BATCH_MATMUL_PAYLOAD,
    lambda: build_assign_schedule("linalg.batch_matmul"),
)

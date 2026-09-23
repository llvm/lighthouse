# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
from lighthouse.execution.target import TargetInfo
from lighthouse.schedule.builders import schedule_boilerplate


def run(name: str, payload_str: str, target_op: str):
    print(f"Test: {name}", flush=True)
    with TargetInfo.override(core_count=4):
        with ir.Context(), ir.Location.unknown():
            lh_dialects.register_and_load()
            payload = ir.Module.parse(payload_str)
            with schedule_boilerplate() as (sched, named_seq):
                ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
                assign_tile_sizes(ops, strategy="cache")
                transform.yield_()
            sched.body.operations[0].apply(payload.operation)
            print(payload)


PAYLOAD_GENERIC = """
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%a: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %e = tensor.empty() : tensor<16x64xf32>
    %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<16x64xf32>)
        outs(%e : tensor<16x64xf32>) {
    ^bb0(%i: f32, %o: f32):
      linalg.yield %i : f32
    } -> tensor<16x64xf32>
    return %g : tensor<16x64xf32>
  }
}
"""


PAYLOAD_MATMUL = """
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


PAYLOAD_PACK = """
module {
  func.func @main(%a: tensor<128x256xf32>) -> tensor<128x256xf32> {
    %d = tensor.empty() : tensor<4x8x32x32xf32>
    %p = linalg.pack %a inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %d
        : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
    %o = tensor.empty() : tensor<128x256xf32>
    %u = linalg.unpack %p inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %o
        : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
    return %u : tensor<128x256xf32>
  }
}
"""


PAYLOAD_ELTWISE_1D = """
module {
    func.func @main(%a: tensor<128xf32>, %b: tensor<128xf32>) -> tensor<128xf32> {
        %e = tensor.empty() : tensor<128xf32>
        %r = linalg.elementwise <add>
                ins(%a, %b : tensor<128xf32>, tensor<128xf32>)
                outs(%e : tensor<128xf32>) -> tensor<128xf32>
        return %r : tensor<128xf32>
    }
}
"""


PAYLOAD_ELTWISE_2D = """
module {
    func.func @main(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
        %e = tensor.empty() : tensor<64x64xf32>
        %sum = linalg.elementwise <add>
                ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
        return %sum : tensor<64x64xf32>
    }
}
"""


# CHECK-LABEL: Test: cache_strategy_generic
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 64>
run("cache_strategy_generic", PAYLOAD_GENERIC, "linalg.generic")

# CHECK-LABEL: Test: cache_strategy_pack
# CHECK: linalg.pack
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 1>
run("cache_strategy_pack", PAYLOAD_PACK, "linalg.pack")

# CHECK-LABEL: Test: cache_strategy_unpack
# CHECK: linalg.unpack
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32>
run("cache_strategy_unpack", PAYLOAD_PACK, "linalg.unpack")

# CHECK-LABEL: Test: cache_strategy_matmul
# CHECK: linalg.matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 32, 32, 0>
run("cache_strategy_matmul", PAYLOAD_MATMUL, "linalg.matmul")

# CHECK-LABEL: Test: cache_strategy_eltwise_1d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 128>
run("cache_strategy_eltwise_1d", PAYLOAD_ELTWISE_1D, "linalg.elementwise")

# CHECK-LABEL: Test: cache_strategy_eltwise_2d
# CHECK: linalg.elementwise
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 16, 16>
run("cache_strategy_eltwise_2d", PAYLOAD_ELTWISE_2D, "linalg.elementwise")

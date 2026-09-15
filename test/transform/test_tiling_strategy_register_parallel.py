# RUN: %PYTHON %s | FileCheck %s

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform.transform_ext import assign_tile_sizes
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


# CHECK-LABEL: Test: register_parallel_strategy
# CHECK: linalg.batch_matmul
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 1, 8, 32, 0>
run("register_parallel_strategy", PAYLOAD, build_schedule)

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


def build_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "linalg.generic")
        assign_tile_sizes(ops, strategy="cache")
        transform.yield_()
    return sched


# CHECK-LABEL: Test: cache_strategy
# CHECK: linalg.generic
# CHECK-SAME: transform_ext.tile_sizes = array<i64: 0, 32>
run("cache_strategy", PAYLOAD, build_schedule)

# RUN: %PYTHON %s | FileCheck %s

"""Tests for the compute_num_threads transform op."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

PAYLOAD = """
module {
  func.func @main() { return }
}
"""


def build_schedule(wg, sg, base):
    i64 = ir.IntegerType.get_signless(64)

    def param(v):
        return transform.ParamConstantOp(
            transform.AnyParamType.get(), ir.IntegerAttr.get(i64, v)
        ).param

    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        nb = transform_ext.compute_num_threads(
            [param(v) for v in wg], [param(v) for v in sg], base=base
        )
        transform.annotate(func, "num_threads", param=nb)
        transform.yield_()
    return sched


def run(name, wg, sg, base):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        sched = build_schedule(wg, sg, base)
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# base * (128 // 32) * (256 // 64) = 16 * 4 * 4 = 256.
# CHECK-LABEL: Test: compute_num_threads
# CHECK: func.func @main() attributes {num_threads = 256 : i64}
run("compute_num_threads", wg=[128, 256], sg=[32, 64], base=16)

# An untiled (0) dim counts as a factor of 1: 1 * (128 // 32) * 1 = 4.
# CHECK-LABEL: Test: compute_num_threads_untiled_dim
# CHECK: func.func @main() attributes {num_threads = 4 : i64}
run("compute_num_threads_untiled_dim", wg=[128, 0], sg=[32, 0], base=1)

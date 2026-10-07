# RUN: %PYTHON %s | FileCheck %s

"""Tests for the compute_sg_layout transform op."""

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


def build_schedule(wg, sg, red, transpose):
    i64 = ir.IntegerType.get_signless(64)

    def params(vals):
        return [
            transform.ParamConstantOp(
                transform.AnyParamType.get(), ir.IntegerAttr.get(i64, v)
            ).param
            for v in vals
        ]

    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        sg_layout, sg_data = transform_ext.compute_sg_layout(
            params(wg),
            params(sg),
            params(red) if red else None,
            transpose=transpose,
        )
        transform.annotate(func, "sg_layout_m", param=sg_layout[0])
        transform.annotate(func, "sg_layout_n", param=sg_layout[1])
        transform.annotate(func, "sg_data_m", param=sg_data[0])
        transform.annotate(func, "sg_data_n", param=sg_data[1])
        transform.yield_()
    return sched


def run(name, wg, sg, red=None, transpose=0):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(PAYLOAD)
        sched = build_schedule(wg, sg, red, transpose)
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# sg_layout = wg // sg = [128 // 32, 256 // 64] = [4, 4]; sg_data = sg = [32, 64].
# CHECK-LABEL: Test: compute_sg_layout
# CHECK-DAG: sg_layout_m = 4 : i64
# CHECK-DAG: sg_layout_n = 4 : i64
# CHECK-DAG: sg_data_m = 32 : i64
# CHECK-DAG: sg_data_n = 64 : i64
run("compute_sg_layout", wg=[128, 256], sg=[32, 64])

# Reduction dim falls back to red // sg (or red) when wg/sg are untiled (0).
# CHECK-LABEL: Test: compute_sg_layout_reduction
# CHECK-DAG: sg_layout_m = 8 : i64
# CHECK-DAG: sg_layout_n = 1 : i64
# CHECK-DAG: sg_data_m = 8 : i64
# CHECK-DAG: sg_data_n = 32 : i64
run("compute_sg_layout_reduction", wg=[64, 0], sg=[8, 0], red=[0, 32])

# transpose reverses the wg dims: wg becomes [256, 128] before dividing.
# CHECK-LABEL: Test: compute_sg_layout_transpose
# CHECK-DAG: sg_layout_m = 8 : i64
# CHECK-DAG: sg_layout_n = 2 : i64
# CHECK-DAG: sg_data_m = 32 : i64
# CHECK-DAG: sg_data_n = 64 : i64
run("compute_sg_layout_transpose", wg=[128, 256], sg=[32, 64], transpose=1)

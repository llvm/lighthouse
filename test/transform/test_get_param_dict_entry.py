# RUN: %PYTHON %s | FileCheck %s

"""Test for the get_param_dict_entry transform op."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

PAYLOAD = """
module {
  func.func @main() {
    return
  }
}
"""


def build_schedule():
    i64 = ir.IntegerType.get_signless(64)
    with schedule_boilerplate() as (sched, named_seq):
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        dict_attr = ir.DictAttr.get(
            {
                "wg_m": ir.IntegerAttr.get(i64, 256),
                "k_tile": ir.IntegerAttr.get(i64, 16),
            }
        )
        dict_param = transform.ParamConstantOp(transform.AnyParamType.get(), dict_attr)
        k_tile = transform_ext.get_param_dict_entry(dict_param.param, "k_tile")
        transform.annotate(func, "k_tile", param=k_tile)
        transform.yield_()
    return sched


def run(name, payload_str):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build_schedule()
        sched.body.operations[0].apply(payload.operation)
        print(payload)


# CHECK-LABEL: Test: get_param_dict_entry
# CHECK: func.func @main() attributes {k_tile = 16 : i64}
run("get_param_dict_entry", PAYLOAD)

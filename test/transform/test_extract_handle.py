# RUN: %PYTHON %s | FileCheck %s

"""Tests for the extract_handle transform op."""

from mlir import ir
from mlir.dialects import transform

import lighthouse.dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate

PAYLOAD = """
module {
  func.func @a() { return }
  func.func @b() { return }
  func.func @c() { return }
}
"""


def build_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        # Positive index picks the first match, negative index the last.
        first = transform_ext.extract_handle(funcs, 0)
        last = transform_ext.extract_handle(funcs, -1)
        transform.annotate(first, "first")
        transform.annotate(last, "last")
        transform.yield_()
    return sched


def build_oob_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        # Out-of-range index with silenceable=True fails without aborting.
        transform_ext.extract_handle(funcs, 5, silenceable=True)
        transform.yield_()
    return sched


def run(name, payload_str, build):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build()
        try:
            sched.body.operations[0].apply(payload.operation)
        except ValueError as error:
            print("failed:", error, flush=True)
            return
        print(payload)


# CHECK-LABEL: Test: extract_handle
# CHECK: func.func @a() attributes {first}
# CHECK: func.func @b() {
# CHECK: func.func @c() attributes {last}
run("extract_handle", PAYLOAD, build_schedule)

# CHECK-LABEL: Test: extract_handle_out_of_range
# CHECK: failed:
run("extract_handle_out_of_range", PAYLOAD, build_oob_schedule)

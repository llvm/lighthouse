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

PARAM_PAYLOAD = """
module {
  func.func @a() attributes {tile = 0 : i64} { return }
  func.func @b() attributes {tile = 64 : i64} { return }
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


def build_param_schedule():
    with schedule_boilerplate() as (sched, named_seq):
        funcs = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        params = transform_ext.get_named_attribute(funcs, "tile")
        for name, index in [
            ("first", 0),
            ("last", 1),
            ("negative", -1),
            ("negative_first", -2),
        ]:
            selected = transform_ext.extract_handle(params, index)
            assert selected.type == params.type
            transform.annotate(named_seq.bodyTarget, name, param=selected)
        index = transform.param_constant(
            transform.AnyParamType.get(),
            ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1),
        )
        selected = transform_ext.extract_handle(params, index)
        transform.annotate(named_seq.bodyTarget, "dynamic", param=selected)
        transform.yield_()
    return sched


def build_param_oob_schedule(index, silenceable=False, empty=False):
    def build():
        with schedule_boilerplate() as (sched, named_seq):
            funcs = lh_transform.match_op(
                named_seq.bodyTarget, "missing.op" if empty else "func.func"
            )
            params = transform_ext.get_named_attribute(funcs, "tile")
            transform_ext.extract_handle(params, index, silenceable=silenceable)
            transform.yield_()
        return sched

    return build


def run(name, payload_str, build, expect_failure=False):
    print(f"Test: {name}", flush=True)
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        payload = ir.Module.parse(payload_str)
        sched = build()
        assert sched.operation.verify()
        try:
            sched.body.operations[0].apply(payload.operation)
        except ValueError as error:
            if not expect_failure:
                raise
            print("failed:", error, flush=True)
            return
        assert not expect_failure, "Expected extraction to fail"
        print(payload)


# CHECK-LABEL: Test: extract_handle
# CHECK: func.func @a() attributes {first}
# CHECK: func.func @b() {
# CHECK: func.func @c() attributes {last}
run("extract_handle", PAYLOAD, build_schedule)

# CHECK-LABEL: Test: extract_handle_out_of_range
# CHECK: failed:
run("extract_handle_out_of_range", PAYLOAD, build_oob_schedule, expect_failure=True)

# CHECK-LABEL: Test: extract_param_handle
# CHECK: {dynamic = 64 : i64, first = 0 : i64, last = 64 : i64, negative = 64 : i64, negative_first = 0 : i64}
run("extract_param_handle", PARAM_PAYLOAD, build_param_schedule)

# CHECK-LABEL: Test: extract_param_out_of_range
# CHECK: failed:
run(
    "extract_param_out_of_range",
    PARAM_PAYLOAD,
    build_param_oob_schedule(2),
    expect_failure=True,
)

# CHECK-LABEL: Test: extract_param_negative_out_of_range
# CHECK: failed:
run(
    "extract_param_negative_out_of_range",
    PARAM_PAYLOAD,
    build_param_oob_schedule(-3),
    expect_failure=True,
)

# CHECK-LABEL: Test: extract_param_out_of_range_silenceable
# CHECK: failed:
run(
    "extract_param_out_of_range_silenceable",
    PARAM_PAYLOAD,
    build_param_oob_schedule(2, silenceable=True),
    expect_failure=True,
)

# CHECK-LABEL: Test: extract_param_empty
# CHECK: failed:
run(
    "extract_param_empty",
    PARAM_PAYLOAD,
    build_param_oob_schedule(0, empty=True),
    expect_failure=True,
)

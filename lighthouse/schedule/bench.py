from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext

from .builders import schedule_boilerplate

# Default name of the benchmarking wrapper function created by
# `bench_wrapper_schedule`.
DEFAULT_BENCH_FUNC_NAME = "__benchmark"


def bench_wrapper_schedule(payload_func: str, bench_name: str = None) -> ir.Module:
    """Build a schedule that wraps ``payload_func`` in a benchmarking function.

    The returned schedule matches the `func.func` named ``payload_func`` and
    wraps it in a benchmarking function named ``bench_name`` (see
    `transform_ext.wrap_in_benching_func`). It must be applied to the payload
    module before any other schedule in an optimizing pipeline.
    """
    if not bench_name:
        bench_name = DEFAULT_BENCH_FUNC_NAME
    with ir.Location.unknown():
        with schedule_boilerplate(result_types=[transform.any_op_t()]) as (
            schedule,
            named_seq,
        ):
            named_func = structured.structured_match(
                transform.AnyOpType.get(),
                target=named_seq.bodyTarget,
                ops={"func.func"},
                op_attrs={"sym_name": ir.StringAttr.get(payload_func)},
            )
            bench_func = transform_ext.wrap_in_benching_func(
                named_func, bench_name=bench_name
            )
            transform.yield_([bench_func])

    schedule.body.operations[0].verify()
    return schedule

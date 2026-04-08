from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from lighthouse.dialects import transform_ext

from .builders import schedule_boilerplate


def convert_function_results(payload_func: str = None) -> ir.Module:
    """
    A schedule that converts the payload function's return values to arguments.

    Args:
        payload_func: The name of the payload function to convert. If None, all
        func.func ops will be converted.
    Returns:
        Schedule
    """
    with ir.Location.unknown():
        with schedule_boilerplate(result_types=[transform.any_op_t()]) as (
            schedule,
            named_seq,
        ):
            matched_func = structured.structured_match(
                transform.AnyOpType.get(),
                target=named_seq.bodyTarget,
                ops={"func.func"},
                op_attrs={"sym_name": ir.StringAttr.get(payload_func)}
                if payload_func
                else None,
            )
            new_func = transform_ext.convert_func_results_to_args(matched_func)
            transform.yield_([new_func])

    schedule.body.operations[0].verify()
    return schedule

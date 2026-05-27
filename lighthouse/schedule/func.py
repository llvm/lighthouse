from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from lighthouse.dialects.transform import transform_ext
import lighthouse.transform as lh_transform

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


def convert_const_resources_to_args(payload_func: str = None) -> ir.Module:
    """
    A schedule that converts all constant resources to arguments.

    Applies CSE and canonicalization to the parent module after conversion.

    Args:
        payload_func: The name of the payload function to convert. If None, all
        func.func ops will be converted.
    Returns:
        Schedule
    """
    with ir.Location.unknown():
        with schedule_boilerplate(result_types=()) as (
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
            mod = transform.get_parent_op(
                transform.AnyOpType.get(),
                matched_func,
                op_name="builtin.module",
                deduplicate=True,
            )
            transform_ext.convert_const_resources_to_args(matched_func)
            lh_transform.cleanup(mod)
            transform.yield_()

    schedule.body.operations[0].verify()
    return schedule

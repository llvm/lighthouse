from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager

from mlir import ir
from mlir.dialects import transform


def create_schedule() -> ir.Module:
    """
    Create a transform schedule module.

    Returns:
        MLIR module
    """
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    return schedule


def create_named_sequence(
    schedule: ir.Module,
    sym_name: str = "__transform_main",
    input_types: Sequence[ir.Type] = [],
    result_types: Sequence[ir.Type] = [],
    is_readonly: bool = False,
) -> transform.NamedSequenceOp:
    """
    Create a named sequence inside a schedule module.

    Args:
        sym_name: Sequence name
        input_types: Input types (default: a single arg)
        result_types: Result types (default: a single return)
        is_readonly: Mark inputs as readonly

    Returns:
        Named transform sequence
    """

    if not input_types:
        input_types = [transform.any_op_t()]
    if not result_types:
        result_types = [transform.any_op_t()]

    arg_attrs = [{"transform.consumed": ir.UnitAttr.get()}]
    if is_readonly:
        arg_attrs = [{"transform.readonly": ir.UnitAttr.get()}]

    with ir.InsertionPoint(schedule.body):
        named_seq = transform.NamedSequenceOp(
            sym_name,
            input_types,
            result_types,
            arg_attrs=arg_attrs,
        )
    return named_seq


@contextmanager
def schedule_boilerplate(
    sym_name: str = "__transform_main",
    input_types: Sequence[ir.Type] = [],
    result_types: Sequence[ir.Type] = [],
    is_readonly: bool = False,
) -> Iterator[tuple[ir.Module, transform.NamedSequenceOp]]:
    """
    Context decorator for creating schedules.

    It creates a new schedule module with an empty transform named sequence.
    The insertion point is automatically placed within the sequence.

    Args:
        sym_name: Sequence name
        input_types: Input types (default: a single arg)
        result_types: Result types (default: a single return)
        is_readonly: Mark inputs as readonly

    Returns:
        Schedule and named transform sequence
    """

    schedule = create_schedule()
    named_sequence = create_named_sequence(
        schedule,
        sym_name=sym_name,
        input_types=input_types,
        result_types=result_types,
        is_readonly=is_readonly,
    )
    with ir.InsertionPoint(schedule.body):
        with ir.InsertionPoint(named_sequence.body):
            yield schedule, named_sequence

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import transform


def create_schedule() -> ir.Module:
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    return schedule


def create_named_sequence(
    schedule: ir.Module,
    sym_name: str = "__transform_main",
    input_types: Sequence[ir.Type] = [],
    result_types: Sequence[ir.Type] = [],
    is_readonly: bool = True,
) -> transform.NamedSequenceOp:
    arg_attrs = None
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

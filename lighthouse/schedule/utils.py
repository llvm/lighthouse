from contextlib import contextmanager
from mlir import ir
from mlir.dialects import transform


@contextmanager
def schedule_boilerplate():
    schedule = ir.Module.create()
    schedule.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(schedule.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [transform.AnyOpType.get()],
            [transform.AnyOpType.get()],
            arg_attrs=[{"transform.consumed": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            yield schedule, named_sequence

from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transforms import linalg_morph_ops
from lighthouse.transforms import cleanup


def schedule_linalg_to_generic_ops() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        linalg_morph_ops(
            named_seq.bodyTarget, named_to_generic=True, category_to_generic=True
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def schedule_linalg_to_category_ops() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        linalg_morph_ops(
            named_seq.bodyTarget, generic_to_category=True, named_to_category=True
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def schedule_linalg_to_named_ops() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        # TODO: Add category to named when it becomes available.
        linalg_morph_ops(
            named_seq.bodyTarget,
            generic_to_named=True,
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import linalg_morph_ops
from lighthouse.transform import cleanup


def linalg_to_generic() -> ir.Module:
    """Morph all linalg ops to generic ops."""
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        linalg_morph_ops(
            named_seq.bodyTarget, named_to_generic=True, category_to_generic=True
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def linalg_to_category() -> ir.Module:
    """Morph all linalg ops to category ops."""
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        linalg_morph_ops(
            named_seq.bodyTarget, generic_to_category=True, named_to_category=True
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def linalg_to_named() -> ir.Module:
    """Morph all linalg ops to named ops."""
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

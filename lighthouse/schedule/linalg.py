from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from .builders import create_schedule
from .builders import create_named_sequence
import lighthouse.transform as lh_transform


def linalg_morph(
    named_to_generic: bool = False,
    named_to_category: bool = False,
    category_to_generic: bool = False,
    generic_to_category: bool = False,
    generic_to_named: bool = False,
) -> ir.Module:
    """
    Morph all linalg ops.

    Args:
        named_to_generic: Named to generic ops
        named_to_category: Named to category ops
        category_to_generic: Category to generic ops
        generic_to_category: Generic to category ops
        generic_to_named: Generic to named ops
    Returns:
        Schedule
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        lh_transform.linalg_morph_ops(
            ops,
            named_to_generic=named_to_generic,
            named_to_category=named_to_category,
            category_to_generic=category_to_generic,
            generic_to_category=generic_to_category,
            generic_to_named=generic_to_named,
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def linalg_to_generic() -> ir.Module:
    """
    Morph all linalg ops to generic ops.

    Returns:
        Schedule
    """
    return linalg_morph(named_to_generic=True, category_to_generic=True)


def linalg_to_category() -> ir.Module:
    """
    Morph all linalg ops to category ops.

    Returns:
        Schedule
    """
    return linalg_morph(generic_to_category=True, named_to_category=True)


def linalg_to_named() -> ir.Module:
    """
    Morph all linalg ops to named ops.

    Returns:
        Schedule
    """
    # TODO: Add category to named when it becomes available.
    return linalg_morph(
        generic_to_named=True,
    )


def linalg_contract_fold_unit_dims() -> ir.Module:
    """
    Fold unit dims of linalg contract.

    NOTE: The rewrite currently applies linalg morphism and folds all generics.

    Returns:
        Schedule
    """
    sched = create_schedule()
    named_seq = create_named_sequence(sched, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        # TODO: Match only contracts when the folding pattern supports them.
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ops = lh_transform.linalg_morph_ops(ops, category_to_generic=True)
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Works only on generics.
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        lh_transform.linalg_morph_ops(ops, generic_to_category=True)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return sched

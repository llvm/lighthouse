from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from .builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def linalg_contract_fold_unit_dims() -> ir.Module:
    """
    Fold unit dims of linalg contract.

    NOTE: The rewrite currently applies linalg morphism and folds all generics.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        # TODO: Match only contracts when the folding pattern supports them.
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        ops = transform.apply_registered_pass(
            transform.any_op_t(),
            ops,
            "linalg-morph-ops",
            options={
                "category-to-generic": True,
            },
        )
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # Works only on generics.
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_registered_pass(
            transform.any_op_t(),
            ops,
            "linalg-morph-ops",
            options={
                "generic-to-category": True,
            },
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

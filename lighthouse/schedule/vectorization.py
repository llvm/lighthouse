from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, vector, tensor

from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def vectorize_linalg() -> ir.Module:
    """
    Vectorize all linalg ops.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(
            named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
        )
        with lh_transform.foreach(ops) as op:
            structured.structured_vectorize(op, [], create_named_contraction=True)
            transform.yield_()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_reduction_to_contract()
            vector.apply_patterns_vector_transfer_permutation_patterns()
            vector.apply_patterns_vector_fold_arith_extension()
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def vectorize_all() -> ir.Module:
    """
    Vectorize all ops.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        structured.structured_vectorize_children_and_apply_patterns(
            transform.any_op_t(), ops
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def x86_vectorization() -> ir.Module:
    """
    Apply x86-specific vector rewrites.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        lh_transform.x86_vector_patterns(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def simplify_vector_ops() -> ir.Module:
    """
    Apply simplification patterns to vector operations.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            # FIXME: This transform breaks AVX512 FMA recognition,
            # but it's in the other sub-schedule of the same name.
            # vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
            tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
            transform.apply_patterns_canonicalization()
        transform.yield_()
    return schedule


def flatten_vector_ops() -> ir.Module:
    """
    Flatten vector ops to 1D.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        lh_transform.flatten_vector_ops(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return schedule

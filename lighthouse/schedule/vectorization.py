from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector

from .builders import create_schedule
from .builders import create_named_sequence
import lighthouse.transform as lh_transform


def vectorize_linalg() -> ir.Module:
    """
    Vectorize all linalg ops.

    Returns:
        Schedule
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        ops = lh_transform.match_op(
            named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
        )
        lh_transform.vectorize_ops(
            ops,
            vectorize_kwargs=dict(create_named_contraction=True),
        )
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
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        lh_transform.vectorize_all_ops(ops)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def x86_vectorization() -> ir.Module:
    """
    Apply x86-specific vector rewrites.

    Returns:
        Schedule
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        lh_transform.x86_vector_patterns(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transforms import vectorize_ops
from lighthouse.transforms import x86_vector_patterns
from lighthouse.transforms import cleanup


def schedule_vectorize_linalg() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        vectorize_ops(
            named_seq.bodyTarget,
            structured.MatchInterfaceEnum.LinalgOp,
            vectorize_kwargs=dict(create_named_contraction=True),
        )
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_reduction_to_contract()
            vector.apply_patterns_vector_transfer_permutation_patterns()
            vector.apply_patterns_vector_fold_arith_extension()
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule


def schedule_x86_vectorization() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        x86_vector_patterns(named_seq.bodyTarget)
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

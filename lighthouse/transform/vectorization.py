from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import x86


def vectorize_ops(
    target,
    vector_sizes: list = [],
    vectorize_kwargs: dict = {},
):
    """
    Apply vectorization to the target.

    Args:
        target: Handle to target
        vector_sizes: Vector sizes
        vectorize_kwargs: Options passed to vectorization transform
    """
    foreach = transform.ForeachOp([], (target,))
    with ir.InsertionPoint(foreach.body):
        op = foreach.bodyTargets[0]
        structured.structured_vectorize(op, vector_sizes, **vectorize_kwargs)
        transform.yield_()


def x86_vector_patterns(target):
    """
    Apply x86-specific vector patterns.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        x86.apply_patterns_x86_vector_contract_to_packed_type_dot_product()
        x86.apply_patterns_x86_vector_contract_to_fma()
        x86.apply_patterns_x86_sink_vector_producer_ops()

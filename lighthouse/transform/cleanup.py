from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import vector


def cleanup(target):
    """
    Apply canonicalization patterns.

    Args:
        target: Handle to target
    """
    transform.apply_cse(target)
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        transform.apply_patterns_canonicalization()


def simplify_vector_ops(target):
    """
    Apply simplification patterns to vector operations.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        vector.apply_patterns_vector_flatten_vector_transfer_ops()
        transform.apply_patterns_canonicalization()

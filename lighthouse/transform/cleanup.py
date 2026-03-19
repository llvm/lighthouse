from mlir import ir
from mlir.dialects import transform


def cleanup(target):
    """
    Apply canonicalization patterns.

    Args:
        target: Handle to target
    """
    transform.apply_cse(target)
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        transform.apply_patterns_canonicalization()

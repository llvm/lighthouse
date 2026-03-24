from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tensor
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
        vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
        tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
        transform.apply_patterns_canonicalization()


def flatten_vector_ops(target):
    """
    Apply dimension flattening patterns to vector operations.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        vector.apply_patterns_vector_flatten_vector_transfer_ops()
        transform.apply_patterns_canonicalization()

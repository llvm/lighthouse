from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.utils.mlir import opview, is_structural_contraction


def is_contraction_op(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is a linalg contraction (matmul-like) op.

    Recognizes both named contractions (e.g. linalg.batch_matmul) and their
    generic form, independent of rank. A contraction contracts a reduction
    dimension shared by two inputs, which distinguishes it from the surrounding
    elementwise or plain (single-input) reduction ops.
    """
    ov = opview(op)
    if "linalg" not in ov.operation.name:
        return False
    return linalg.isa_contraction_op(ov) or is_structural_contraction(ov)


FilterContractionOpsOp = make_filter_handles_op(
    "filter_contraction_ops", is_contraction_op
)


def filter_contraction_ops(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterContractionOpsOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the contraction-op subset of `target`.
    """
    return FilterContractionOpsOp(target=target).ops

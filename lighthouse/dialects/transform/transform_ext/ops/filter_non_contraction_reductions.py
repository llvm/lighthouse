from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.utils.mlir import is_linalg_reduction_op

FilterNonContractionReductionsOp = make_filter_handles_op(
    "filter_non_contraction_reductions", is_linalg_reduction_op
)


def filter_non_contraction_reductions(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a FilterNonContractionReductionsOp.

    Keeps single-output linalg reductions (e.g. row max/sum, norms) that are
    not contractions, convolutions or pooling ops.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the non-contraction reduction subset of `target`.
    """
    return FilterNonContractionReductionsOp(target=target).ops

from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)


def _has_reduction_loop(op: ir.OpView) -> bool:
    """Check whether a (generic) op has at least one reduction iterator type."""
    build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
    parallel = build(linalg.IteratorType.parallel, context=op.context)
    return any(it != parallel for it in op.iterator_types)


def is_reduction_op(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is a linalg op with at least one reduction dimension."""
    ov = op.opview if isinstance(op, ir.Operation) else op
    if not hasattr(ov, "iterator_types"):
        return False
    return _has_reduction_loop(ov)


FilterReductionOpsOp = make_filter_handles_op("filter_reduction_ops", is_reduction_op)


def filter_reduction_ops(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterReductionOpsOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the reduction-op subset of `target`.
    """
    return FilterReductionOpsOp(target=target).ops

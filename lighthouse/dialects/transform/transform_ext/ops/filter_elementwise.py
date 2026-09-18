from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)
from lighthouse.utils.mlir import indexing_maps, linalg_inputs
from lighthouse.utils.mlir import opview, is_linalg_all_loops_parallel


def is_elementwise(op: ir.Operation | ir.OpView) -> bool:
    """Check whether the op is an elementwise linalg op.

    NOTE: Mimics corresponding Linalg util as it is not exposed
          in the Python bindings yet.
    """
    ov = opview(op)
    if ov.operation.name == "linalg.elementwise":
        return True
    maps = indexing_maps(ov)
    if maps is None:
        return False
    if isinstance(ov, linalg.GenericOp) and not is_linalg_all_loops_parallel(ov):
        return False
    if not all(m.is_projected_permutation for m in maps):
        return False
    num_inputs = len(linalg_inputs(ov))
    return all(m.is_permutation for m in maps[num_inputs:])


FilterElementwiseOp = make_filter_handles_op("filter_elementwise", is_elementwise)


def filter_elementwise(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a FilterElementwiseOp.

    Args:
        target: Handle to target op(s).
    Returns:
        Handle to the elementwise subset of `target`.
    """
    return FilterElementwiseOp(target=target).ops

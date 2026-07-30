from mlir import ir
from mlir.dialects import transform, linalg

from lighthouse.dialects.transform.transform_ext.ops.filter_handles import (
    make_filter_handles_op,
)


def _parse_num_loops(params) -> int | None:
    if len(params) == 1 and isinstance(params[0], ir.IntegerAttr):
        return params[0].value
    return None


def _has_at_least_num_loops(op: ir.Operation | ir.OpView, num_loops: int) -> bool:
    if "linalg" not in op.name:
        return False
    if hasattr(op, "indexing_maps"):
        map: ir.AffineMap = op.indexing_maps[0].value
        return map.n_dims >= num_loops
    if hasattr(op, "iterator_types"):
        return len(op.iterator_types) >= num_loops
    if isinstance(op.opview, linalg.FillOp):
        return op.outputs[0].type.rank >= num_loops
    return False


FilterNumLoopsOp = make_filter_handles_op(
    "filter_num_loops",
    _has_at_least_num_loops,
    parse_param=_parse_num_loops,
)


def filter_num_loops(
    target: ir.Value[transform.AnyOpType],
    num_loops: int | ir.Value[transform.AnyParamType],
) -> ir.Value:
    """
    snake_case wrapper to create a FilterNumLoopsOp.

    Args:
        target: Handle to target op
        num_loops: Number of loops to filter by
    Returns:
        List of matching ops that have at least `num_loops` loops.
    """
    if isinstance(num_loops, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), num_loops)
        num_loops = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return FilterNumLoopsOp(target=target, param=num_loops).ops

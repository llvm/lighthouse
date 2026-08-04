from collections.abc import Sequence

from mlir import ir

from lighthouse.utils.mlir import (
    opview,
    indexing_maps,
    dim_position,
    linalg_inputs,
    linalg_outputs,
)
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa


def is_propagatable(op: ir.Operation | ir.OpView) -> bool:
    """Check whether tile sizes may be propagated onto this op.

    True for any structured linalg op that is not a fusion barrier; non-linalg
    ops have no indexing maps to translate tiles through and are excluded.
    """
    return indexing_maps(op) is not None and not fa.is_fusion_barrier(op)


def _map_for_value(
    op: ir.OpView, value: ir.Value, maps: Sequence[ir.AffineMap]
) -> ir.AffineMap | None:
    """Return the indexing map associated with `value` on `op`."""
    inputs = linalg_inputs(op)
    outputs = linalg_outputs(op)
    if inputs is None or outputs is None:
        return None
    for i, operand in enumerate(inputs):
        if operand == value:
            return maps[i]
    for k, operand in enumerate(outputs):
        if operand == value:
            return maps[len(inputs) + k]
    # A result corresponds (positionally) to an output operand of a DPS op.
    for r, result in enumerate(op.results):
        if result == value:
            return maps[len(inputs) + r]
    return None


def tiles_on_value(
    op: ir.Operation | ir.OpView,
    sizes: Sequence[int],
    value: ir.Value,
) -> list[int] | None:
    """Per-dimension tiles that `op`, tiled by `sizes`, induces on `value`.

    Returns one entry per dimension of `value` (tensor-dim order); 0 means the
    dimension is left untiled / not constrained by `op`. Returns None when `op`
    is not a structured linalg op or does not touch `value`.

    Projecting both a producer's and a consumer's sizes onto the shared tensor
    this way makes tile comparisons robust to transposition: differently ordered
    iteration spaces still agree when they tile the shared tensor identically.
    """
    ov = opview(op)
    maps = indexing_maps(ov)
    if maps is None:
        return None
    value_map = _map_for_value(ov, value, maps)
    if value_map is None:
        return None
    tiles = [0] * ir.ShapedType(value.type).rank
    for tensor_dim, expr in enumerate(value_map.results):
        pos = dim_position(expr)
        if pos is not None and pos < len(sizes):
            tiles[tensor_dim] = sizes[pos]
    return tiles


def compatible_on_value(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    dst_op: ir.Operation | ir.OpView,
    dst_sizes: Sequence[int],
    shared: ir.Value,
) -> bool:
    """Check whether two ops tile a shared tensor compatibly.

    The tiles each op induces on `shared` are compared per tensor dimension. A
    dimension only conflicts when both ops tile it with *different* non-zero
    sizes; a zero (untiled / broadcast / unconstrained) side is a wildcard and
    never conflicts.

    Returns True when the tiles cannot be determined, so grouping errs toward
    fusion rather than over-splitting.
    """
    a = tiles_on_value(src_op, src_sizes, shared)
    b = tiles_on_value(dst_op, dst_sizes, shared)
    if a is None or b is None:
        return True
    return all(x == y for x, y in zip(a, b) if x != 0 and y != 0)


def propagate_through_value(
    src_op: ir.Operation | ir.OpView,
    src_sizes: Sequence[int],
    shared: ir.Value,
    dst_op: ir.Operation | ir.OpView,
) -> list[int] | None:
    """Propagate tile sizes from `src_op` to `dst_op` via a shared tensor.

    The shared tensor's per-dimension tiles are derived from `src_op`'s sizes and
    mapped onto `dst_op`'s iteration space; reduction dims of `dst_op` stay untiled.

    Returns `dst_op`'s tile sizes (loop order), or None if not possible.
    """
    src = opview(src_op)
    dst = opview(dst_op)

    dst_maps = indexing_maps(dst)
    if dst_maps is None:
        return None

    # Tile size per dimension of the shared tensor, as induced by the source.
    tensor_tiles = tiles_on_value(src, src_sizes, shared)
    dst_map = _map_for_value(dst, shared, dst_maps)
    if tensor_tiles is None or dst_map is None:
        return None

    if len(list(dst.results)) != 1:
        return None
    dst_out_map = dst_maps[-1]
    dst_parallel = {
        pos for pos in (dim_position(e) for e in dst_out_map.results) if pos is not None
    }

    dst_sizes = [0] * dst_out_map.n_dims
    for tensor_dim, expr in enumerate(dst_map.results):
        pos = dim_position(expr)
        if pos is None:
            continue
        # Only tile parallel dims of the consumer; leave reductions untiled.
        if pos in dst_parallel:
            dst_sizes[pos] = tensor_tiles[tensor_dim]
    return dst_sizes

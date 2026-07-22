from collections.abc import Sequence

from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import opview, indexing_maps, dim_position, linalg_outputs

# Attribute used to annotate payload ops with their target tile sizes
# (one entry per iteration dimension, in loop order).
TILE_SIZES_ATTR_NAME = "transform_ext.tile_sizes"

# Default size used for tiled dimensions when no hint is provided.
DEFAULT_TILE_SIZE = 32

# Number of innermost parallel dimensions tiled with the full tile size.
# Other parallel dimensions are tiled with unit size to keep them sequential.
DEFAULT_PARALLEL_TILE_DIMS = 2


def _output_tensor_dim_of_iter_dim(out_map: ir.AffineMap) -> dict[int, int]:
    """Map each iteration dim to the output tensor dim it indexes (if any)."""
    mapping: dict[int, int] = {}
    for tensor_dim, expr in enumerate(out_map.results):
        pos = dim_position(expr)
        if pos is not None:
            mapping[pos] = tensor_dim
    return mapping


def _disable_small_tiles(
    op: ir.OpView,
    out_map: ir.AffineMap,
    sizes: list[int],
    tile_size: int,
) -> None:
    """Disable tiling for parallel dims whose static extent is below tile_size.

    Tiling a dimension that is smaller than the tile size only adds loop
    overhead, so such dimensions are left untiled (size 0).
    """
    out_type = ir.ShapedType(linalg_outputs(op)[0].type)
    iter_to_tensor = _output_tensor_dim_of_iter_dim(out_map)
    for iter_dim, tensor_dim in iter_to_tensor.items():
        if sizes[iter_dim] <= 1:
            # Unit / untiled dimensions are always fine.
            continue
        dim = out_type.shape[tensor_dim]
        if ir.ShapedType.is_static_size(dim) and dim < tile_size:
            sizes[iter_dim] = 0


def compute_tile_sizes(
    op: ir.Operation | ir.OpView,
    tile_size: int = DEFAULT_TILE_SIZE,
    parallel_tile_dims: int = DEFAULT_PARALLEL_TILE_DIMS,
) -> list[int] | None:
    """Compute target tile sizes for an op over its iteration space.

    For structured ops, the innermost `parallel_tile_dims` parallel (output) dims
    are tiled with `tile_size`, any remaining parallel dims (batch / outer M/N)
    with a unit size, and reduction dims are left untiled. Statically small
    parallel dims are also left untiled.

    pack / unpack ops are tiled to a single inner tile per iteration: a pack tiles
    every source dim by 1; an unpack tiles its packed output dims (`inner_dims_pos`)
    by the inner tile size and the rest by 1.

    Returns one size per iteration dim (loop order), or None if unsupported.
    """
    ov = opview(op)

    # pack / unpack have no affine indexing maps; their tiling follows the pack
    # structure (they are tiled standalone, not fused into a group).
    if isinstance(ov, linalg.PackOp):
        return [1] * ir.ShapedType(ov.source.type).rank
    if isinstance(ov, linalg.UnPackOp):
        sizes = [1] * ir.ShapedType(ov.result.type).rank
        inner_dims = ir.DenseI64ArrayAttr(ov.inner_dims_pos)
        inner_tiles = ir.DenseI64ArrayAttr(ov.static_inner_tiles)
        for dim, tile in zip(inner_dims, inner_tiles):
            sizes[dim] = tile
        return sizes

    maps = indexing_maps(ov)
    if maps is None:
        return None
    # Only single-output ops are supported for now.
    if len(linalg_outputs(ov)) != 1:
        return None

    # The output operand's map is the last one (inputs first, then outputs).
    out_map = maps[-1]
    sizes = [0] * out_map.n_dims

    # Tile the innermost parallel dims with the full tile size and
    # any outer parallel dims (batch, outer M/N) with a unit size.
    parallel_dims = [
        pos for pos in (dim_position(e) for e in out_map.results) if pos is not None
    ]
    if not parallel_dims:
        return None
    for d in parallel_dims[:-parallel_tile_dims]:
        sizes[d] = 1
    for d in parallel_dims[-parallel_tile_dims:]:
        sizes[d] = tile_size

    _disable_small_tiles(ov, out_map, sizes, tile_size)
    return sizes


def get_tile_sizes_attr(op: ir.Operation | ir.OpView) -> list[int] | None:
    """Return the tile sizes annotated on an op, or None if not annotated."""
    attr = opview(op).operation.attributes
    if TILE_SIZES_ATTR_NAME not in attr:
        return None
    return list(ir.DenseI64ArrayAttr(attr[TILE_SIZES_ATTR_NAME]))


def set_tile_sizes_attr(op: ir.Operation | ir.OpView, sizes: Sequence[int]) -> None:
    """Annotate an op with its target tile sizes."""
    operation = opview(op).operation
    operation.attributes[TILE_SIZES_ATTR_NAME] = ir.DenseI64ArrayAttr.get(list(sizes))


def clear_tile_sizes_attr(op: ir.Operation | ir.OpView) -> None:
    """Remove the tile-size annotation from an op, if present."""
    attrs = opview(op).operation.attributes
    if TILE_SIZES_ATTR_NAME in attrs:
        del attrs[TILE_SIZES_ATTR_NAME]

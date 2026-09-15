from collections.abc import Sequence

from mlir import ir

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import opview

from .tiling import (
    StrategyContext,
    get_tiling_strategy,
)

# Attribute used to annotate payload ops with their target tile sizes
# (one entry per iteration dimension, in loop order).
TILE_SIZES_ATTR_NAME = "transform_ext.tile_sizes"

# Default size used for tiled dimensions when no hint is provided.
DEFAULT_TILE_SIZE = 32


def compute_tile_sizes(
    op: ir.Operation | ir.OpView,
    tile_size: int = DEFAULT_TILE_SIZE,
    strategy: str = "cache",
    target: TargetInfo | None = None,
) -> list[int] | None:
    """Compute per-dimension tile sizes for an operation's iteration space.

    Args:
        op: Payload operation to analyze.
        tile_size: Tiling size hint.
        strategy: Tiling strategy.
        target: Target-hardware description.

    Returns:
        A list of tile sizes (one per iteration dimension), or `None` when the
        operation is unsupported by the selected strategy.
    """
    ctx = StrategyContext(tile_size=tile_size, target=target)
    return get_tiling_strategy(strategy).compute(op, ctx)


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


__all__ = [
    "DEFAULT_TILE_SIZE",
    "TILE_SIZES_ATTR_NAME",
    "clear_tile_sizes_attr",
    "compute_tile_sizes",
    "get_tile_sizes_attr",
    "set_tile_sizes_attr",
]

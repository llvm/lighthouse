from mlir import ir
from mlir.dialects import linalg

from lighthouse.utils.mlir import opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import disable_small_tiles, parallel_and_reduction_dims


class CacheTilingStrategy(TilingStrategy):
    """Cache-level tiling.

    Intended as a first-level tiling.
    Improves memory access patterns and helps expose parallelism.
    """

    _PARALLEL_TILE_DIMS = 2

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        ov = opview(op)

        # pack / unpack have no affine indexing maps; their tiling follows
        # the pack structure.
        if isinstance(ov, linalg.PackOp):
            return [1] * ir.ShapedType(ov.source.type).rank
        if isinstance(ov, linalg.UnPackOp):
            sizes = [1] * ir.ShapedType(ov.result.type).rank
            inner_dims = ir.DenseI64ArrayAttr(ov.inner_dims_pos)
            inner_tiles = ir.DenseI64ArrayAttr(ov.static_inner_tiles)
            for dim, tile in zip(inner_dims, inner_tiles):
                sizes[dim] = tile
            return sizes

        out_map = self.output_map(ov)
        if out_map is None:
            return None

        sizes = [0] * out_map.n_dims
        parallel_dims, _ = parallel_and_reduction_dims(out_map)
        if not parallel_dims:
            return None

        for d in parallel_dims[: -self._PARALLEL_TILE_DIMS]:
            sizes[d] = 1
        for d in parallel_dims[-self._PARALLEL_TILE_DIMS :]:
            sizes[d] = ctx.tile_size
        disable_small_tiles(ov, out_map, sizes, ctx.tile_size)
        return sizes

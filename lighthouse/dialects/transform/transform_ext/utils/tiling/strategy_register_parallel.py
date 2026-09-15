from mlir import ir

from lighthouse.utils.mlir import opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_parallel_tiles,
    disable_small_tiles,
    parallel_and_reduction_dims,
)
from .target_caps import (
    generic_parallel_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
)


class RegisterParallelTilingStrategy(TilingStrategy):
    """Register-level tiling of parallel dimensions; target-derived defaults."""

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        out_map = self.output_map(op)
        if out_map is None:
            return None

        sizes = [0] * out_map.n_dims
        parallel_dims, _ = parallel_and_reduction_dims(out_map)
        if not parallel_dims:
            return None

        ov = opview(op)
        if is_amx_bf16_contraction(ov, ctx.target):
            inner_tiles = [32, 32]
        elif is_f32_contraction(ov):
            inner_tiles = [8, 32]
        else:
            inner_tiles = generic_parallel_tiles(ov, out_map, ctx.target)

        assign_parallel_tiles(parallel_dims, inner_tiles, sizes)
        disable_small_tiles(ov, out_map, sizes, max(inner_tiles, default=ctx.tile_size))
        return sizes

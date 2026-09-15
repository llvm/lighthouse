from mlir import ir

from lighthouse.utils.mlir import opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_reduction_tiles,
    parallel_and_reduction_dims,
)
from .target_caps import (
    generic_reduction_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
)


class RegisterReductionTilingStrategy(TilingStrategy):
    """Register-level tiling of reduction dimensions; target-derived defaults."""

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        out_map = self.output_map(op)
        if out_map is None:
            return None

        sizes = [0] * out_map.n_dims
        _, reduction_dims = parallel_and_reduction_dims(out_map)
        if not reduction_dims:
            return None

        ov = opview(op)
        if is_amx_bf16_contraction(ov, ctx.target):
            red_tiles = [32]
        elif is_f32_contraction(ov):
            red_tiles = [2]
        else:
            red_tiles = generic_reduction_tiles()

        assign_reduction_tiles(reduction_dims, red_tiles, sizes)
        return sizes

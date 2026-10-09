from mlir import ir

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import is_linalg_reduction_op, opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_parallel_tiles,
    assign_reduction_tiles,
    largest_multiple_divisor,
    parallel_and_reduction_dims,
)
from .reduction_analysis import reduction_info
from .strategy_register_parallel import EltwiseRegisterTiling
from .target_caps import (
    generic_parallel_tiles,
    generic_reduction_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
)


class RegisterUnrollTilingStrategy(TilingStrategy):
    """Register-level unroll-friendly tiling; target-derived defaults."""

    @staticmethod
    def _reduction_op_tiles(
        op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Unrolled shape of a non-contraction reduction.
        One vector of an outer reduction, or a whole row of an inner one.
        """
        info = reduction_info(op)
        if info is None:
            return None
        sizes = [1] * len(info.dim_sizes)
        lanes = EltwiseRegisterTiling.lane_count(target, info.elem_type)
        if info.inner:
            red_dim_size = info.dim_sizes[info.vector_dim]
            if red_dim_size is not None and red_dim_size < lanes:
                return None
            # Keep a single wide horizontal reduction instead of a serial chain.
            sizes[info.vector_dim] = 0
        else:
            tile = largest_multiple_divisor(
                info.dim_sizes[info.vector_dim], lanes, lanes
            )
            if tile is None:
                return None
            sizes[info.vector_dim] = tile
        return sizes

    def compute(
        self, op: ir.Operation | ir.OpView, ctx: StrategyContext
    ) -> list[int] | None:
        out_map = self.output_map(op)
        if out_map is None:
            return None

        ov = opview(op)
        # Reductions may have no parallel dims left (e.g. a rank-1 row reduction).
        if is_linalg_reduction_op(ov):
            tiles = self._reduction_op_tiles(ov, ctx.target)
            if tiles is not None:
                return tiles

        sizes = [0] * out_map.n_dims
        parallel_dims, reduction_dims = parallel_and_reduction_dims(out_map)
        if not parallel_dims:
            return None

        if is_amx_bf16_contraction(ov, ctx.target):
            par_tiles = [16, 16]
            red_tiles = [32]
        elif is_f32_contraction(ov):
            par_tiles = [1, 16]
            red_tiles = [1]
        else:
            par_tiles = generic_parallel_tiles(ov, out_map, ctx.target)
            red_tiles = generic_reduction_tiles()

        assign_parallel_tiles(parallel_dims, par_tiles, sizes)
        assign_reduction_tiles(reduction_dims, red_tiles, sizes)
        return sizes

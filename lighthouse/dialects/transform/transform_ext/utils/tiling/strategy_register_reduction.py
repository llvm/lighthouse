from mlir import ir

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import is_linalg_reduction_op, opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_reduction_tiles,
    largest_multiple_divisor,
    largest_divisor,
    parallel_and_reduction_dims,
)
from .reduction_analysis import reduction_info
from .strategy_register_parallel import EltwiseRegisterTiling
from .target_caps import (
    generic_reduction_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
    reduction_acc_chains,
)


class RegisterReductionTilingStrategy(TilingStrategy):
    """Register-level tiling of reduction dimensions; target-derived defaults."""

    # Reduced rows per register tile of an outer reduction, amortizing the
    # loop overhead over several lane-wise accumulations.
    _OUTER_REDUCTION_TILE = 2

    @classmethod
    def _reduction_op_tiles(
        cls, op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Tiles of a non-contraction reduction; parallel dims are left untiled.

        Inner reductions are tiled by lanes x accumulator chains along the
        reduced vector dim; outer ones tile their innermost reduced dim.
        """
        info = reduction_info(op)
        if info is None:
            return None
        dim_sizes = info.dim_sizes
        sizes = [0] * len(dim_sizes)
        for d in info.reduction_dims:
            sizes[d] = 1

        if info.inner:
            lanes = EltwiseRegisterTiling.lane_count(target, info.elem_type)
            limit = lanes * reduction_acc_chains(target)
            dim_size = dim_sizes[info.vector_dim]
            if dim_size is None or dim_size < lanes:
                return None
            # Short enough for one horizontal reduction: keep it whole.
            if dim_size <= limit:
                sizes[info.vector_dim] = 0
            else:
                tile = largest_multiple_divisor(dim_size, limit, lanes)
                if tile is None:
                    return None
                sizes[info.vector_dim] = tile
        else:
            innermost = info.reduction_dims[-1]
            sizes[innermost] = largest_divisor(
                dim_sizes[innermost], cls._OUTER_REDUCTION_TILE
            )
        if not any(sizes):
            return None
        return sizes

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
        elif is_linalg_reduction_op(ov):
            return self._reduction_op_tiles(ov, ctx.target)
        else:
            red_tiles = generic_reduction_tiles()

        assign_reduction_tiles(reduction_dims, red_tiles, sizes)
        return sizes

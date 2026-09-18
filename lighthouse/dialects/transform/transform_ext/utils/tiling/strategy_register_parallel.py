from mlir import ir


from lighthouse.execution.target import RegisterInfo, TargetInfo
from lighthouse.utils.mlir import linalg_outputs, opview, is_linalg_eltwise_op

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


class EltwiseRegisterTiling:
    """Target-aware register-bank tiling for elementwise ops."""

    @staticmethod
    def register_bank_tile_count(target: TargetInfo | None, elem_type: ir.Type) -> int:
        """Return the number of scalar elements that fit inside the target SIMD register bank."""
        register = (target and target.vector_register_info()) or RegisterInfo(
            width_bits=512, count=32
        )
        if isinstance(elem_type, ir.FloatType):
            # Assumes that native sub-32bit float computation is not supported.
            elem_bits = max(32, elem_type.width)
        elif isinstance(elem_type, ir.IntegerType):
            elem_bits = elem_type.width
        else:
            elem_bits = 32
        return max(1, (register.width_bits * register.count) // elem_bits)

    @staticmethod
    def _tile_from_parallel_dims(
        parallel_dims: list[int], shape: list[int], tile_size: int
    ) -> list[int]:
        """
        Spread a register-bank-sized tile across the trailing parallel axes
        without exceeding shape extents.
        """
        assert len(parallel_dims) == len(shape), (
            f"parallel dims {parallel_dims} do not match output rank {len(shape)}"
        )

        tiles = [1] * len(parallel_dims)
        if not tiles:
            return tiles

        remaining = max(1, tile_size)
        for axis_index in reversed(range(len(parallel_dims))):
            extent = shape[axis_index]
            if ir.ShapedType.is_dynamic_size(extent):
                tile = remaining
            else:
                tile = min(extent, remaining)
            tiles[axis_index] = max(1, tile)
            remaining = max(1, remaining // max(1, tiles[axis_index]))
        return tiles

    @classmethod
    def choose_parallel_tile_shape(
        cls,
        op: ir.Operation | ir.OpView,
        parallel_dims: list[int],
        target: TargetInfo | None,
    ) -> list[int]:
        """
        Choose elementwise parallel tiles from the target register bank
        and the output-shape footprint.
        """
        if not parallel_dims:
            return []

        out_type = ir.ShapedType(linalg_outputs(op)[0].type)
        out_elem = out_type.element_type
        register_bank = cls.register_bank_tile_count(target, out_elem)
        return cls._tile_from_parallel_dims(
            parallel_dims, list(out_type.shape), register_bank
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
        elif is_linalg_eltwise_op(ov):
            inner_tiles = EltwiseRegisterTiling.choose_parallel_tile_shape(
                ov, parallel_dims, ctx.target
            )
        else:
            inner_tiles = generic_parallel_tiles(ov, out_map, ctx.target)

        assign_parallel_tiles(parallel_dims, inner_tiles, sizes)
        # Keep the intended register-bank footprint on the trailing parallel axes, but
        # only prune dims that are below the smallest non-unit tile we actually emitted.
        guard_tile = min(
            (tile for tile in inner_tiles if tile > 1), default=ctx.tile_size
        )
        disable_small_tiles(ov, out_map, sizes, guard_tile)
        return sizes

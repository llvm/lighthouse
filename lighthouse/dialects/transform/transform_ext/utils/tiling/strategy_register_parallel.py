from mlir import ir


from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import (
    is_linalg_eltwise_op,
    is_linalg_reduction_op,
    linalg_outputs,
    opview,
)

from .strategy_base import StrategyContext, TilingStrategy
from .common import (
    assign_parallel_tiles,
    disable_small_tiles,
    largest_multiple_divisor,
    largest_divisor,
    parallel_and_reduction_dims,
)
from .reduction_analysis import reduction_info
from .target_caps import (
    generic_parallel_tiles,
    is_amx_bf16_contraction,
    is_f32_contraction,
    reduction_acc_chains,
    register_info,
)


class EltwiseRegisterTiling:
    """Target-aware register-bank tiling for elementwise ops."""

    @staticmethod
    def compute_bits(elem_type: ir.Type) -> int:
        """Bit width of elementwise arithmetic on `elem_type`.

        Assumes no native sub-32-bit float arithmetic, so such floats are
        computed as f32. This also holds for the lane-wise combiners of
        reductions but not for contractions or other specialized instructions
        (e.g. AMX or AVX512-BF16 dot products).
        """
        if isinstance(elem_type, ir.FloatType):
            return max(32, elem_type.width)
        if isinstance(elem_type, ir.IntegerType):
            return elem_type.width
        return 32

    @classmethod
    def lane_count(cls, target: TargetInfo | None, elem_type: ir.Type) -> int:
        """SIMD lanes of one register for elementwise arithmetic on `elem_type`."""
        return max(1, register_info(target).width_bits // cls.compute_bits(elem_type))

    @classmethod
    def register_bank_tile_count(
        cls, target: TargetInfo | None, elem_type: ir.Type
    ) -> int:
        """Return the number of scalar elements that fit inside the target SIMD register bank."""
        return cls.lane_count(target, elem_type) * register_info(target).count

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

    @staticmethod
    def _reduction_op_tiles(
        op: ir.Operation | ir.OpView, target: TargetInfo | None
    ) -> list[int] | None:
        """Tiles of a non-contraction reduction; reduction dims are left untiled.

        Parallel dims are tiled to keep lanes x accumulator chains independent
        accumulators busy: inner reductions tile several rows together, outer
        ones vectorize the contiguous parallel dim and spread chains over outer
        parallel dims.
        """
        info = reduction_info(op)
        if info is None or not info.parallel_dims:
            return None
        dim_sizes = info.dim_sizes
        lanes = EltwiseRegisterTiling.lane_count(target, info.elem_type)
        chains = reduction_acc_chains(target)
        sizes = [0] * len(dim_sizes)
        for d in info.parallel_dims:
            sizes[d] = 1

        if info.inner:
            red_dim_size = dim_sizes[info.vector_dim]
            # A reduced dim shorter than a vector (e.g. a pooling window) has
            # no lanes to fill: leave the op to its neighbours' tiles.
            if red_dim_size is None or red_dim_size < lanes:
                return None
            unroll = max(1, min(chains, red_dim_size // lanes))
            rows_dim = info.parallel_dims[-1]
            sizes[rows_dim] = largest_divisor(
                dim_sizes[rows_dim], max(1, chains // unroll)
            )
            return sizes

        vec_tile = largest_multiple_divisor(
            dim_sizes[info.vector_dim], lanes * chains, lanes
        )
        if vec_tile is None:
            return None
        sizes[info.vector_dim] = vec_tile
        # Spread missing accumulator chains over the next outer parallel dims.
        remaining = max(1, chains // max(1, vec_tile // lanes))
        for d in reversed([p for p in info.parallel_dims if p != info.vector_dim]):
            if remaining <= 1:
                break
            sizes[d] = largest_divisor(dim_sizes[d], remaining)
            remaining = max(1, remaining // sizes[d])
        return sizes

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
        elif is_linalg_reduction_op(ov):
            return self._reduction_op_tiles(ov, ctx.target)
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

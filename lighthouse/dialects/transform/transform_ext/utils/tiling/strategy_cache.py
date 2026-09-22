from mlir import ir
from mlir.dialects import linalg

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import linalg_outputs, opview, is_linalg_eltwise_op

from .strategy_base import StrategyContext, TilingStrategy
from .common import disable_small_tiles, parallel_and_reduction_dims
from .target_caps import vector_lane_count


class EltwiseCacheTiling:
    """Heuristic for first-level cache tiling of all-parallel elementwise ops."""

    # Cache tiling parameters.
    # TODO: Use these as fallback and derive them from target info when available.
    _TARGET_BYTES_L1 = 8 * 1024
    _MIN_TILE_BYTES = 4 * 1024
    _TILES_PER_CORE = 4
    # Default tile size for dynamic dimension.
    _DEFAULT_DYN_DIM = 256
    # Cap number of rows in case of large pow2 strides to minimize cache conflicts.
    _MAX_ROWS_POW2 = 8
    # Value representing a dynamic dimension size.
    _DYNAMIC = ir.ShapedType.get_dynamic_size()

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        """Round `value` up to the next multiple of `multiple`."""
        if value <= 0 or multiple <= 0:
            return value if value > 0 else multiple
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _row_major_strides(shape: list[int], dtype_size: int) -> list[int]:
        """Return row-major byte strides for the given shape.

        Once a dynamic dimension is seen, strides for all axes to its left
        are unknown too, since they depend on that unresolved extent.
        """
        strides: list[int] = [EltwiseCacheTiling._DYNAMIC] * len(shape)
        stride = dtype_size
        for axis in reversed(range(len(shape))):
            dim = shape[axis]
            if ir.ShapedType.is_dynamic_size(dim):
                break
            strides[axis] = stride
            stride = stride * dim
        return strides

    @staticmethod
    def _is_pow2_large(value: int) -> bool:
        """Return True for large power-of-two strides that should be capped."""
        return (
            value != EltwiseCacheTiling._DYNAMIC
            and value >= 4096
            and value & (value - 1) == 0
        )

    @staticmethod
    def _tile_bytes(tile: list[int], planning_shape: list[int], dtype_size: int) -> int:
        """Approximate the bytes in the candidate cache tile.

        ``planning_shape`` must already have dynamic dims replaced with concrete sizes.
        """
        total = dtype_size
        for axis in range(len(planning_shape)):
            extent = planning_shape[axis]
            block = tile[axis]
            if block <= 0:
                return 0
            total *= min(block, extent)
        return total

    @classmethod
    def _grow_to_floor(
        cls, tile: list[int], planning_shape: list[int], dtype_size: int
    ) -> list[int]:
        """Grow the tile until it reaches the minimum footprint."""
        for _ in range(len(planning_shape) * 8):
            if cls._tile_bytes(tile, planning_shape, dtype_size) >= cls._MIN_TILE_BYTES:
                return tile
            grown = False
            for axis in range(len(planning_shape) - 1, -1, -1):
                dim = planning_shape[axis]
                if tile[axis] >= dim:
                    continue
                tile[axis] = min(dim, max(1, tile[axis] * 2))
                grown = True
                if (
                    cls._tile_bytes(tile, planning_shape, dtype_size)
                    >= cls._MIN_TILE_BYTES
                ):
                    return tile
            if not grown:
                break
        return tile

    @classmethod
    def _check_parallelism(
        cls, tile: list[int], planning_shape: list[int], num_cores: int
    ) -> list[int]:
        """Scale the tile to respect the available parallelism budget."""
        target_tiles = num_cores * cls._TILES_PER_CORE

        def tile_count(axis: int) -> int:
            dim = planning_shape[axis]
            return max(1, (dim + max(1, tile[axis]) - 1) // max(1, tile[axis]))

        counts = [tile_count(axis) for axis in range(len(planning_shape))]
        total_tiles = 1
        for count in counts:
            total_tiles *= count
        if total_tiles >= target_tiles:
            return tile
        for axis in sorted(
            [a for a in range(len(planning_shape) - 1) if tile[a] > 1],
            key=lambda a: tile[a],
            reverse=True,
        ):
            while total_tiles < target_tiles and tile[axis] > 1:
                tile[axis] = max(1, tile[axis] // 2)
                total_tiles //= counts[axis]
                counts[axis] = tile_count(axis)
                total_tiles *= counts[axis]
            if total_tiles >= target_tiles:
                break
        return tile

    @classmethod
    def choose_parallel_tile_shape(
        cls, op: ir.OpView, target: TargetInfo | None
    ) -> list[int]:
        """Return a cache-first tile shape for an all-parallel elementwise op."""
        out_type = ir.ShapedType(linalg_outputs(op)[0].type)
        shape = list(out_type.shape)
        rank = len(shape)
        if rank == 0:
            return []

        elem_width = getattr(out_type.element_type, "width", 8)
        dtype_size = max(1, (elem_width + 7) // 8)

        planning_shape = [
            cls._DEFAULT_DYN_DIM if ir.ShapedType.is_dynamic_size(dim) else dim
            for dim in shape
        ]
        strides = cls._row_major_strides(shape, dtype_size)
        # Row-major layout: the innermost axis is always fastest-varying.
        fast_axis = rank - 1

        tiles = [1] * rank
        vector_width = vector_lane_count(target, out_type.element_type)
        fast_extent = planning_shape[fast_axis]
        candidate_fast = max(
            vector_width,
            cls._round_up(max(1, fast_extent // 4), vector_width),
        )
        tiles[fast_axis] = min(fast_extent, candidate_fast)

        remaining_elems = max(
            1,
            cls._TARGET_BYTES_L1 // max(1, dtype_size) // max(1, tiles[fast_axis]),
        )
        for axis in sorted(
            [a for a in range(rank) if a != fast_axis], key=lambda a: -a
        ):
            if remaining_elems <= 1:
                tiles[axis] = 1
                continue
            take = min(planning_shape[axis], remaining_elems)
            tiles[axis] = take
            remaining_elems = max(1, remaining_elems // max(1, take))

        for axis in [a for a in range(rank) if a != fast_axis]:
            stride = strides[axis]
            if stride == cls._DYNAMIC or cls._is_pow2_large(stride):
                tiles[axis] = min(tiles[axis], cls._MAX_ROWS_POW2)

        for axis in range(rank):
            if not ir.ShapedType.is_dynamic_size(shape[axis]):
                tiles[axis] = min(tiles[axis], shape[axis])

        tiles = cls._grow_to_floor(tiles, planning_shape, dtype_size)
        target_cores = (target or TargetInfo.host()).core_count()
        tiles = cls._check_parallelism(tiles, planning_shape, target_cores)
        return tiles


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
        if is_linalg_eltwise_op(ov):
            out_map = self.output_map(ov)
            if out_map is None:
                return None
            sizes = [0] * out_map.n_dims
            tile = EltwiseCacheTiling.choose_parallel_tile_shape(ov, ctx.target)
            for dim, value in enumerate(tile):
                sizes[dim] = value
            disable_small_tiles(ov, out_map, sizes, ctx.tile_size)
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

from mlir import ir
from mlir.dialects import linalg

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.mlir import linalg_outputs, opview

from .strategy_base import StrategyContext, TilingStrategy
from .common import disable_small_tiles, parallel_and_reduction_dims
from .target_caps import vector_lane_count


class EltwiseCacheTiling:
    """Heuristic for first-level cache tiling of all-parallel elementwise ops."""

    _TARGET_BYTES_L1 = 8 * 1024
    _MIN_TILE_BYTES = 4 * 1024
    _DEFAULT_DYN_DIM = 256
    _MAX_ROWS_POW2 = 8
    _TILES_PER_CORE = 4
    _DYNAMIC = ir.ShapedType.get_dynamic_size()

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        """Round `value` up to the next multiple of `multiple`."""
        if value <= 0 or multiple <= 0:
            return value if value > 0 else multiple
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _row_major_strides(shape: list[int], dtype_size: int) -> list[int]:
        """Return row-major byte strides for the given shape."""
        strides: list[int] = [0] * len(shape)
        stride = dtype_size
        for axis in reversed(range(len(shape))):
            dim = shape[axis]
            if ir.ShapedType.is_dynamic_size(dim):
                strides[axis] = EltwiseCacheTiling._DYNAMIC
                continue
            strides[axis] = stride
            stride = stride * dim
        return strides

    @staticmethod
    def _contiguous_axis(strides: list[int], dtype_size: int) -> tuple[int, bool]:
        """Choose the most contiguous axis as the fast tile axis."""
        for axis in range(len(strides) - 1, -1, -1):
            s = strides[axis]
            if s == EltwiseCacheTiling._DYNAMIC:
                return axis, False
            if s == dtype_size:
                return axis, True
        if not strides:
            return 0, False
        return len(strides) - 1, False

    @staticmethod
    def _is_pow2_large(value: int) -> bool:
        """Return True for large power-of-two strides that should be capped."""
        return (
            value != EltwiseCacheTiling._DYNAMIC
            and value >= 4096
            and value & (value - 1) == 0
        )

    @staticmethod
    def _tile_bytes(tile: list[int], shape: list[int], dtype_size: int) -> int:
        """Approximate the bytes in the candidate cache tile."""
        total = dtype_size
        for axis in range(len(shape)):
            extent = shape[axis]
            block = tile[axis]
            if block <= 0:
                return 0
            if ir.ShapedType.is_dynamic_size(extent):
                extent = EltwiseCacheTiling._DEFAULT_DYN_DIM
            total *= min(block, extent)
        return total

    @classmethod
    def _grow_to_floor(
        cls, tile: list[int], shape: list[int], dtype_size: int, fast_axis: int
    ) -> list[int]:
        """Grow the tile until it reaches the minimum L1 footprint."""
        for _ in range(len(shape) * 8):
            if cls._tile_bytes(tile, shape, dtype_size) >= cls._MIN_TILE_BYTES:
                return tile
            grown = False
            for axis in [fast_axis] + [
                a for a in range(len(shape) - 1, -1, -1) if a != fast_axis
            ]:
                dim = shape[axis]
                if ir.ShapedType.is_dynamic_size(dim):
                    dim = cls._DEFAULT_DYN_DIM
                if tile[axis] >= dim:
                    continue
                tile[axis] = min(dim, max(1, tile[axis] * 2))
                grown = True
                if cls._tile_bytes(tile, shape, dtype_size) >= cls._MIN_TILE_BYTES:
                    return tile
            if not grown:
                break
        return tile

    @classmethod
    def _check_parallelism(
        cls, tile: list[int], shape: list[int], fast_axis: int, num_cores: int
    ) -> list[int]:
        """Scale the tile to respect the available parallelism budget."""
        target_tiles = num_cores * cls._TILES_PER_CORE
        total_tiles = 1
        for axis, dim in enumerate(shape):
            if ir.ShapedType.is_dynamic_size(dim):
                continue
            total_tiles *= max(
                1,
                (dim + max(1, tile[axis]) - 1) // max(1, tile[axis]),
            )
        if total_tiles >= target_tiles:
            return tile
        for axis in sorted(
            [a for a in range(len(shape)) if a != fast_axis and tile[a] > 1],
            key=lambda a: tile[a],
            reverse=True,
        ):
            while total_tiles < target_tiles and tile[axis] > 1:
                tile[axis] = max(1, tile[axis] // 2)
                total_tiles = 1
                for dim_axis, dim in enumerate(shape):
                    if ir.ShapedType.is_dynamic_size(dim):
                        continue
                    total_tiles *= max(
                        1,
                        (dim + max(1, tile[dim_axis]) - 1) // max(1, tile[dim_axis]),
                    )
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

        dtype_size = max(1, getattr(out_type.element_type, "width", 8) // 8)
        if isinstance(out_type.element_type, ir.FloatType):
            dtype_size = max(1, (out_type.element_type.width + 7) // 8)
        elif isinstance(out_type.element_type, ir.IntegerType):
            dtype_size = max(1, (out_type.element_type.width + 7) // 8)

        planning_shape = [
            cls._DEFAULT_DYN_DIM if ir.ShapedType.is_dynamic_size(dim) else dim
            for dim in shape
        ]
        strides = cls._row_major_strides(planning_shape, dtype_size)
        fast_axis, _ = cls._contiguous_axis(strides, dtype_size)

        tile = [1] * rank
        vector_width = vector_lane_count(target, out_type.element_type)
        fast_extent = planning_shape[fast_axis]
        candidate_fast = max(
            vector_width,
            cls._round_up(max(1, fast_extent // 4), vector_width),
        )
        tile[fast_axis] = min(fast_extent, candidate_fast)

        remaining_elems = max(
            1,
            cls._TARGET_BYTES_L1 // max(1, dtype_size) // max(1, tile[fast_axis]),
        )
        for axis in sorted(
            [a for a in range(rank) if a != fast_axis], key=lambda a: -a
        ):
            if remaining_elems <= 1:
                tile[axis] = 1
                continue
            take = min(planning_shape[axis], remaining_elems)
            tile[axis] = take
            remaining_elems = max(1, remaining_elems // max(1, take))

        for axis in [a for a in range(rank) if a != fast_axis]:
            stride = strides[axis]
            if stride == cls._DYNAMIC or cls._is_pow2_large(stride):
                tile[axis] = min(tile[axis], cls._MAX_ROWS_POW2)

        for axis in range(rank):
            if not ir.ShapedType.is_dynamic_size(shape[axis]):
                tile[axis] = min(tile[axis], shape[axis])

        tile = cls._grow_to_floor(tile, planning_shape, dtype_size, fast_axis)
        target_cores = (target or TargetInfo.host()).core_count()
        tile = cls._check_parallelism(tile, planning_shape, fast_axis, target_cores)
        return tile


class CacheTilingStrategy(TilingStrategy):
    """Cache-level tiling.

    Intended as a first-level tiling.
    Improves memory access patterns and helps expose parallelism.
    """

    _PARALLEL_TILE_DIMS = 2

    @staticmethod
    def _all_loops_parallel(op: ir.OpView) -> bool:
        """Return True when all iterator types are parallel."""
        build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
        parallel = build(linalg.IteratorType.parallel, context=op.context)
        return all(it == parallel for it in op.iterator_types)

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
        if ov.operation.name == "linalg.elementwise" or (
            isinstance(ov, linalg.GenericOp) and self._all_loops_parallel(ov)
        ):
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

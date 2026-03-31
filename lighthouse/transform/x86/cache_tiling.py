from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects import transform_ext


def matmul_cache_tiling(
    target,
    num_tiles: int,
    tile_size: int = 32,
    fuse_producers: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """
    Applies cache tiling to the target matmul operation.
    Creates a forall loop on successful rewrite.

    This tiling step improves computation's memory access pattern
    and exposes parallelism.

    Optionally, fusion can be performed after tiling to minimize
    data transfers.

    Args:
        target: Handle to target operation.
        num_tiles: Number of expected tile handles.
            Currently, it has to match target op's number of iterators.
            TODO: Remove when tiling ops accept variadic tiles handle.
        tile_size: Target size for tile dimensions.
        fuse_producers: Apply extra producer ops fusion after tiling.
    Returns:
        Handles to the tiled op and created loop
    """
    tiles = transform_ext.get_tiling_sizes(target, tile_dim=tile_size)
    tile_sizes = transform.split_handle(
        results_=[transform.AnyParamType.get()] * num_tiles, handle=tiles
    )
    if fuse_producers:
        # Tile the target and greedily fuse its producers.
        tiled_op, forall_op = structured.FuseOp(
            target,
            tile_sizes=tile_sizes,
            apply_cleanup=True,
            use_forall=True,
        ).results
    else:
        # Only tile the target.
        tiled_op, forall_op = structured.TileUsingForallOp(
            target, sizes=tile_sizes
        ).results
    # TODO: Fuse elementwise consumers.

    return tiled_op, forall_op

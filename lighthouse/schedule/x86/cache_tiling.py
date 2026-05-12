from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def matmul_cache_tiling(
    target: str, tile_size: int = 32, fuse_producers: bool = False
) -> ir.Module:
    """
    Applies cache tiling to the target matmul operation.
    Creates a forall loop on successful rewrite.

    This tiling step improves computation's memory access pattern
    and exposes parallelism.

    Optionally, fusion can be performed after tiling to minimize
    data transfers.

    Args:
        target: Handle to target operation.
        tile_size: Target size for tile dimensions.
        fuse_producers: Apply extra producer ops fusion after tiling.
    Returns:
        Schedule module
    """
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, target)
        with lh_transform.foreach(ops) as op:
            tiles = transform_ext.get_tiling_sizes(op, tile_dim=tile_size)
            if fuse_producers:
                # Tile the target and greedily fuse its producers.
                structured.FuseOp(
                    op,
                    tile_sizes=tiles,
                    apply_cleanup=True,
                    use_forall=True,
                ).results
            else:
                # Only tile the target.
                structured.TileUsingForallOp(op, tile_sizes=tiles).results
            transform.yield_()
        transform.yield_()

    # TODO: Fuse elementwise consumers.
    return sched

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform.structured import MatchInterfaceEnum

from .builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def tile(
    target_op: str | list[str] | MatchInterfaceEnum,
    tile_sizes: list[int],
    fuse_producers: bool = False,
    tile_interchange: list[int] | None = None,
    peel_loops: list[int] = [],
    unroll_factors: list[int] = [],
) -> ir.Module:
    """
    Tile all matching op.

    Optionally, producer fusion can be applied to each tiled op.
    Optionally, peeling or unrolling can be applied to created loops.

    Args:
        target_op: Ops to be matched.
        tile_sizes: Tile sizes.
            The sizes are applied in order of the target loops.
            A tile size of zero implies no tiling for that loop.
            If there are fewer tiles than the number of loops,
            the inner loops are not tiled.
            See underlying transform ops for further details.
        fuse_producers: Tile target and greedily fuse its producers
        tile_interchange: Loop interchange after tiling
        peel_loops: List of loops to peel.
            Loops are peeled in the given order.
            Skipped if None. Exclusive with unrolling.
        unroll_factors: Unroll factors for each loop.
            Unrolling is applied from the innermost loop.
            Skipped if None. Exclusive with peeling.
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
        lh_transform.tile_ops(
            ops,
            tile_sizes=tile_sizes,
            fuse_producers=fuse_producers,
            tile_interchange=tile_interchange,
            peel_loops=peel_loops,
            unroll_factors=unroll_factors,
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

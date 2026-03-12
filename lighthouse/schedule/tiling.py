from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import tile_ops
from lighthouse.transform import cleanup


def tile(
    target_op: str,
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
        target_op: Ops to be matched
        tile_sizes: Tile sizes
        fuse_producers: Tile an op and greedily fuse its producers
        tile_interchange: Loop interchange after tiling
        peel_loops: List of loops to peel.
            Loops are peeled in the given order.
            Skipped if None. Exclusive with unrolling.
        unroll_factors: Unroll factors for each loop.
            Unrolling is applied from the innermost loop.
            Skipped if None. Exclusive with peeling.
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        tile_ops(
            named_seq.bodyTarget,
            target_op,
            tile_sizes=tile_sizes,
            fuse_producers=fuse_producers,
            tile_interchange=tile_interchange,
            peel_loops=peel_loops,
            unroll_factors=unroll_factors,
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

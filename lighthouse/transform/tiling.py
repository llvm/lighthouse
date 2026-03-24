from mlir.dialects import transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured

from lighthouse.transform import foreach


def tile_ops(
    target,
    tile_sizes: list[int],
    fuse_producers: bool = False,
    tile_interchange: list[int] | None = None,
    peel_loops: list[int] = [],
    unroll_factors: list[int] = [],
):
    """
    Apply tiling to the target.

    Optionally, producer fusion can be applied to each tiled op.
    Optionally, peeling or unrolling can be applied to created loops.

    Args:
        target: Handle to target.
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
        unroll_factors: Unroll factors for each loop (same order as loops).
            Zero factor means no unrolling is performed.
            Unrolling is applied from the innermost loop.
            Skipped if None. Exclusive with peeling.
    """
    assert not (len(peel_loops) and len(unroll_factors)), (
        "Both unrolling and peeling is not supported"
    )

    with foreach(target) as op:
        if fuse_producers:
            _, *loops = structured.FuseOp(
                op,
                tile_sizes=tile_sizes,
                tile_interchange=tile_interchange,
                apply_cleanup=True,
            ).results
        else:
            _, *loops = structured.TileUsingForOp(
                op, sizes=tile_sizes, interchange=tile_interchange
            ).results
        for idx in peel_loops:
            loop.LoopPeelOp(
                transform.any_op_t(),
                transform.any_op_t(),
                loops[idx],
                peel_front=False,
                fail_if_already_divisible=False,
            )
        for idx, factor in enumerate(reversed(unroll_factors)):
            if factor == 0:
                continue
            loop.loop_unroll(loops[-1 - idx], factor)
        transform.yield_()

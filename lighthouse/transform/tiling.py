from typing import Sequence

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop
from mlir.dialects.transform import structured


def tile(
    target,
    tile_sizes: list[int],
    fuse_producers: bool = False,
    tile_interchange: list[int] | None = None,
    peel_loops: list[int] = [],
    unroll_factors: list[int] = [],
) -> tuple[ir.Value, Sequence[ir.Value], Sequence[ir.Value]]:
    """
    Apply tiling to the target operation.

    Optionally, producer fusion can be applied to the tiled op.
    Optionally, peeling or unrolling can be applied to created loops.

    Note: unrolling invalidates loop handles.

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
    Returns:
        Handles to:
            - tiled op
            - created tile loops
            - remainder loops after peeling
        The order of the remainder loops corresponds to the tile loops.
    """
    assert not (len(peel_loops) and len(unroll_factors)), (
        "Both unrolling and peeling is not supported"
    )

    if fuse_producers:
        tiled_op, *loops = structured.FuseOp(
            target,
            tile_sizes=tile_sizes,
            tile_interchange=tile_interchange,
            apply_cleanup=True,
        ).results
    else:
        tiled_op, *loops = structured.TileUsingForOp(
            target, sizes=tile_sizes, interchange=tile_interchange
        ).results

    remainder_loops = [None] * len(loops)
    for idx in peel_loops:
        main, partial = loop.LoopPeelOp(
            transform.any_op_t(),
            transform.any_op_t(),
            loops[idx],
            peel_front=False,
            fail_if_already_divisible=False,
        )
        loops[idx] = main
        remainder_loops[idx] = partial

    for idx, factor in enumerate(reversed(unroll_factors)):
        if factor == 0:
            continue
        loop.loop_unroll(loops[-1 - idx], factor)

    return tiled_op, loops, remainder_loops

from mlir import ir

import lighthouse.schedule as lh_schedule

# FIXME: These functions should receive u-Arch and data type parameters and come
# up with the appropriate tiling and unrolling factors based on the target hardware


def matmul_register_tiling(
    target: str,
    tile_size: int = 32,
    reg_tile_batch: int = 0,
    reg_tile_m: int = 8,
    reg_tile_n: int = 32,
    reg_tile_k: int = 2,
) -> ir.Module:
    """
    Applies register tiling to the target matmul operation.

    This tiling step prepares the IR for the x86 vectorization passes.

    Args:
        target: Target operation.
        tile_size: Tile size used in the previous cache tiling step.
        reg_tile_batch: Target size for batch dimension tile.
        reg_tile_m: Target size for M dimension tile.
        reg_tile_n: Target size for N dimension tile.
        reg_tile_k: Target size for K dimension tile.
        batch: True if the input has batch dimension.
    Returns:
        Schedule
    """
    tile_sizes = [reg_tile_m, reg_tile_n, reg_tile_k]
    tile_interchange = []
    if reg_tile_batch:
        tile_sizes = [reg_tile_batch] + tile_sizes
        tile_interchange = [1, 2, 0, 3]

    reg_peel_loops = []
    assert tile_size % reg_tile_k == 0, "Invalid K dim register tiling"
    if tile_size % reg_tile_n != 0:
        reg_peel_loops.append(1)
    if tile_size % reg_tile_m != 0:
        reg_peel_loops.append(0)
    return lh_schedule.tile_ops(
        target_op=target,
        tile_sizes=tile_sizes,
        tile_interchange=tile_interchange,
        peel_loops=reg_peel_loops,
    )


def matmul_register_unroll(
    target: str,
    reg_tile_m: int = 8,
    reg_tile_n: int = 32,
    reg_tile_k: int = 2,
    reg_unroll_m: int = 1,
    reg_unroll_n: int = 16,
    reg_unroll_k: int = 1,
    batch: bool = False,
) -> ir.Module:
    """
    Applies register unrolling to the target matmul operation.

    This unrolling step prepares the IR for the x86 vectorization passes.
    Ensure that shapes are compatible with target hardware instructions.

    Args:
        target: Target operation.
        tile_size: Tile size used in the previous cache tiling step.
        reg_tile_m: Target size for M dimension tile.
        reg_tile_n: Target size for N dimension tile.
        reg_tile_k: Target size for K dimension tile.
        reg_unroll_m: Unroll M dimension after tiling.
        reg_unroll_n: Unroll N dimension after tiling.
        reg_unroll_k: Unroll K dimension after tiling.
        batch: True if the input has batch dimension.
    Returns:
        Schedule
    """
    tile_sizes = [reg_unroll_m, reg_unroll_n, reg_unroll_k]
    if batch:
        tile_sizes = [0] + tile_sizes

    reg_unroll_factors = [
        reg_tile_m // reg_unroll_m,
        reg_tile_n // reg_unroll_n,
        reg_tile_k // reg_unroll_k,
    ]
    return lh_schedule.tile_ops(
        target_op=target,
        tile_sizes=tile_sizes,
        unroll_factors=reg_unroll_factors,
    )

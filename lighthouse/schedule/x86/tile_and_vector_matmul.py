from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector

from lighthouse.schedule import schedule_boilerplate
import lighthouse.transform as lh_transform


def create_schedule(
    options: dict = {},
    tile_sizes: tuple[int, int] = [32, 32],
    register_tile: tuple[int, int, int] = [8, 32, 1],
    matmul_op: str = "linalg.matmul",
) -> ir.Module:
    """
    Specialized one-shot schedule for zero initialized Linalg 2D matrix
    multiplication lowering into sequence of FMA instructions.

    The schedule targets sequence:
        %zero_fill = linalg.fill %zero
        %res = linalg.matmul ins(%matA, %matB) outs(%zero_fill)

    Tiling and vectorization is progressively applied to
    achieve SIMD code generation.

    Args:
        tile_size: Shape for matmul output tiling (M, N)
        register_tile: Shape for register-level sub-tiling (M, N, K)
        matmul_op: Name of target matmul operation
    Returns:
        MLIR transform module.
    """
    if len(tile_sizes) != 2:
        raise ValueError(f"Expected 2 tile sizes but got {len(tile_sizes)}")
    if len(register_tile) != 3:
        raise ValueError(f"Expected 3 block factors but got {len(register_tile)}")

    with schedule_boilerplate() as (schedule, named_seq):
        # GEMM tiling.
        matmul = lh_transform.match_op(named_seq.bodyTarget, [matmul_op])
        with lh_transform.foreach(matmul) as op:
            lh_transform.tile(op, tile_sizes=tile_sizes, fuse_producers=True)
            transform.yield_()

        # Tile buffer initialization for better vectorization.
        tiled_fill = lh_transform.match_op(named_seq.bodyTarget, ["linalg.fill"]).result
        reg_fill = structured.TileUsingForOp(tiled_fill, sizes=[1]).results[0]

        # Register tiling.
        reg_peel_loops = []
        if tile_sizes[1] % register_tile[1] != 0:
            reg_peel_loops.append(1)
        if tile_sizes[0] % register_tile[0] != 0:
            reg_peel_loops.append(0)
        matmul = lh_transform.match_op(named_seq.bodyTarget, [matmul_op])
        with lh_transform.foreach(matmul) as op:
            lh_transform.tile(
                op,
                tile_sizes=register_tile,
                peel_loops=reg_peel_loops,
            )
            transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)

        # Register unroll.
        matmul = lh_transform.match_op(named_seq.bodyTarget, [matmul_op])
        reg_unroll_factors = [
            register_tile[0],
            0,
            register_tile[2],
        ]
        with lh_transform.foreach(matmul) as op:
            lh_transform.tile(
                op,
                tile_sizes=[1, register_tile[1], 1],
                unroll_factors=reg_unroll_factors,
            )
            transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)

        # Vectorize operations.
        matmul = lh_transform.match_op(named_seq.bodyTarget, [matmul_op])
        structured.structured_vectorize(matmul, [], create_named_contraction=True)
        structured.structured_vectorize(reg_fill, [])
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_reduction_to_contract()
            vector.apply_patterns_vector_transfer_permutation_patterns()
        lh_transform.cleanup(named_seq.bodyTarget)

        # Loop hoisting.
        all_loops = lh_transform.match_op(
            named_seq.bodyTarget,
            structured.MatchInterfaceEnum.LoopLikeInterface,
        )
        lh_transform.loop_hoisting(all_loops)

        # Vector cleanup.
        lh_transform.simplify_vector_ops(named_seq.bodyTarget)

        # Lower to broadcast+FMA instructions.
        lh_transform.vector_contract_to_fma(named_seq.bodyTarget)

        # Cleanup vector ops.
        lh_transform.flatten_vector_ops(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()

    schedule.body.operations[0].verify()
    return schedule

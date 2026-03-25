from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import loop
from mlir.dialects.transform import vector
from mlir.dialects.transform import tensor

from lighthouse.schedule import schedule_boilerplate
from lighthouse import transform as lh_transform


def lower_packs_for_vectorization(
    pack_ops,
    pack_tile_sizes: list[int],
    vector_tile_sizes: list[int] | None = None,
    vector_unroll_factors: list[int] = [],
):
    """
    Lower packs into hardware-friendly operations.

    Args:
        pack_ops: Handle to pack operations
        pack_tile_sizes: Pack sub-tiling sizes
        vector_tile_sizes: Target vector shapes
        vector_unroll_factors: Unroll factors for each vector loop.
    """
    with lh_transform.foreach(pack_ops) as pack_op:
        tiled_pack = structured.TileUsingForOp(
            pack_op, sizes=pack_tile_sizes
        ).tiled_linalg_op
        _, _, transpose = structured.structured_lower_pack(
            transform.OperationType.get("tensor.pad"),
            transform.OperationType.get("tensor.expand_shape"),
            transform.OperationType.get("linalg.transpose"),
            tiled_pack,
            lower_pad_like_with_insert_slice=False,
        )
        if vector_tile_sizes:
            _, *loops = structured.TileUsingForOp(
                transpose, sizes=vector_tile_sizes
            ).results
            for idx, factor in enumerate(reversed(vector_unroll_factors)):
                loop.loop_unroll(loops[-1 - idx], factor)
        transform.yield_()


def lower_unpacks_for_vectorization(
    unpack_ops,
    unpack_tile_sizes: list[int],
    vector_tile_sizes: list[int] | None = None,
):
    """
    Lower unpacks into hardware-friendly operations.

    Args:
        unpack_ops: Handle to unpack operations
        unpack_tile_sizes: Unpack sub-tiling sizes
        vector_tile_sizes: Target vector shapes
    """
    with lh_transform.foreach(unpack_ops) as unpack_op:
        tiled_unpack = structured.TileUsingForOp(
            unpack_op, sizes=unpack_tile_sizes
        ).tiled_linalg_op
        if vector_tile_sizes:
            tiled_unpack = structured.TileUsingForOp(
                tiled_unpack, sizes=vector_tile_sizes
            ).tiled_linalg_op
        structured.structured_lower_unpack(
            transform.OperationType.get("tensor.empty"),
            transform.OperationType.get("linalg.transpose"),
            transform.OperationType.get("tensor.collapse_shape"),
            transform.OperationType.get("tensor.extract_slice"),
            transform.OperationType.get("linalg.copy"),
            tiled_unpack,
            lower_unpad_like_with_extract_slice=True,
        )
        transform.yield_()


def lower_packs_unpacks(tile_size: int) -> ir.Module:
    """
    Lower pack and unpack ops into hardware-friendly shapes.

    Args:
        tile_size: Target shape for sub-tiling pack and unpack ops' inner tiles
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        pack_unpack_vector_m = max(8, tile_size)
        pack_unpack_vector_n = min(64, tile_size)
        packs = lh_transform.match_op(named_seq.bodyTarget, "linalg.pack")
        lower_packs_for_vectorization(
            packs,
            pack_tile_sizes=[1, 1],
            vector_tile_sizes=[1, 1, pack_unpack_vector_m, pack_unpack_vector_n],
            vector_unroll_factors=[
                tile_size // pack_unpack_vector_n,
            ],
        )
        lh_transform.cleanup(named_seq.bodyTarget)

        unpacks = lh_transform.match_op(named_seq.bodyTarget, "linalg.unpack")
        lower_unpacks_for_vectorization(
            unpacks,
            unpack_tile_sizes=[tile_size, tile_size],
            vector_tile_sizes=[1],
        )
        transposes = lh_transform.match_op(named_seq.bodyTarget, "linalg.transpose")
        with lh_transform.foreach(transposes) as tranpose:
            structured.structured_vectorize(tranpose, [])
            transform.yield_()

        # Cleanup.
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            tensor.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers()
            transform.apply_patterns_canonicalization()
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            vector.apply_patterns_vector_cast_away_vector_leading_one_dim()
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

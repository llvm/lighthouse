from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import vector
from mlir.dialects.transform import tensor

from lighthouse.schedule import schedule_boilerplate
from lighthouse import transform as lh_transform
from lighthouse.dialects.transform import transform_ext


def _vectorize_innermost(ops):
    """
    Tile each op's leading dims into unit loops and vectorize the innermost dim.

    Rank-agnostic: yields N-1 unit sizes for an N-dim op, so the outer dims
    become loops and the innermost dim is vectorized to a 1-D vector.

    Args:
        ops: Handle to structured linalg ops (e.g. transposes / copies).
    """
    with lh_transform.foreach(ops) as op:
        sizes = transform_ext.get_leading_unit_tile_sizes(op)
        # Tiling with scf.forall as scf.for tiling API does not accept a single
        # handle with all sizes.
        # Ultimately, the behavior is the same. Leaving it as is for now.
        tiled = structured.TileUsingForallOp(op, tile_sizes=sizes).tiled_op
        structured.structured_vectorize(tiled, [])
        transform.yield_()


def lower_packs(pack_ops):
    """
    Tile, lower and vectorize pack ops.

    Tiling adapts to arbitrary pack dimensionality.
    Ops produced by the lowering are vectorized in place.

    Args:
        pack_ops: Handle to (tile-size annotated) pack operations.
    """
    with lh_transform.foreach(pack_ops) as pack_op:
        tile_sizes = transform_ext.get_tile_sizes(pack_op)
        tiled_pack = structured.TileUsingForallOp(
            pack_op, tile_sizes=tile_sizes
        ).tiled_op
        lowered = structured.LowerPackOp(
            transform.OperationType.get("tensor.pad"),
            transform.OperationType.get("tensor.expand_shape"),
            transform.OperationType.get("linalg.transpose"),
            tiled_pack,
            lowerPadLikeWithInsertSlice=False,
        )
        _vectorize_innermost(lowered.transpose_op)
        transform.yield_()


def lower_unpacks(unpack_ops):
    """
    Tile, lower and vectorize unpack ops.

    Tiling adapts to arbitrary unpack dimensionality.
    Ops produced by the lowering are vectorized in place.

    Args:
        unpack_ops: Handle to (tile-size annotated) unpack operations.
    """
    with lh_transform.foreach(unpack_ops) as unpack_op:
        tile_sizes = transform_ext.get_tile_sizes(unpack_op)
        tiled_unpack = structured.TileUsingForallOp(
            unpack_op, tile_sizes=tile_sizes
        ).tiled_op
        lowered = structured.LowerUnPackOp(
            transform.OperationType.get("tensor.empty"),
            transform.OperationType.get("linalg.transpose"),
            transform.OperationType.get("tensor.collapse_shape"),
            transform.OperationType.get("tensor.extract_slice"),
            transform.OperationType.get("linalg.copy"),
            tiled_unpack,
            lowerUnpadLikeWithExtractSlice=True,
        )
        _vectorize_innermost(lowered.transpose_op)
        _vectorize_innermost(lowered.copy_op)
        transform.yield_()


def lower_packs_unpacks(tile_size: int = 32) -> ir.Module:
    """
    Lower pack and unpack ops into hardware-friendly, vectorized ops.

    Args:
        tile_size: Target size for sub-tiling pack and unpack ops' inner tiles
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        packs = lh_transform.match_op(named_seq.bodyTarget, "linalg.pack")
        lower_packs(transform_ext.assign_tile_sizes(packs, tile_size=tile_size))
        lh_transform.cleanup(named_seq.bodyTarget)

        unpacks = lh_transform.match_op(named_seq.bodyTarget, "linalg.unpack")
        lower_unpacks(transform_ext.assign_tile_sizes(unpacks, tile_size=tile_size))

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

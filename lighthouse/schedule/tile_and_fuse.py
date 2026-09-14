from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.dialects.transform import loop

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


# Named GEMM ops treated as default anchors.
# Their tiles take precedence and drive the tiling of surrounding ops.
_GEMM_ANCHOR_OPS = [
    "linalg.matmul",
    "linalg.matmul_transpose_a",
    "linalg.matmul_transpose_b",
    "linalg.batch_matmul",
    "linalg.batch_reduce_matmul",
    "linalg.contract",
    "linalg.matvec",
    "linalg.vecmat",
    "linalg.batch_matvec",
]


def assign_and_propagate_tile_sizes(
    anchor_op: str | list[str] | None = None,
    tile_size: int = 32,
    strategy: str = "cache",
    propagate: bool = True,
    propagate_through_loops: bool = False,
) -> ir.Module:
    """
    Assign tile sizes to anchor ops and propagate them to their neighbors.

    Anchors ops are annotated and their sizes propagated to neighboring ops
    so a fusable group shares one tiling. Can be applied repeatedly with different
    anchors (e.g. GEMMs first, then leftover elementwise ops); already-annotated
    ops keep their sizes, so earlier assignments win.

    Args:
        anchor_op: Op(s) to anchor tiling on. Defaults to the GEMM op family.
        tile_size: Tiling size hint.
        strategy: Tiling strategy.
        propagate: Whether to propagate the tile sizes to neighboring ops.
        propagate_through_loops: Whether to propagate through loop-carried
            values instead of treating loops as barriers (default: False).
    Returns:
        Schedule
    """
    if anchor_op is None:
        anchor_op = _GEMM_ANCHOR_OPS

    with schedule_boilerplate() as (sched, named_seq):
        anchors = lh_transform.match_op(named_seq.bodyTarget, anchor_op)
        annotated = transform_ext.assign_tile_sizes(
            anchors,
            tile_size=tile_size,
            strategy=strategy,
        )
        if propagate:
            transform_ext.propagate_tile_sizes(
                annotated, propagate_through_loops=propagate_through_loops
            )
        transform.yield_()
    return sched


def assign_elementwise_tile_sizes(
    tile_size: int = 32,
    strategy: str = "cache",
    propagate: bool = True,
    propagate_through_loops: bool = False,
) -> ir.Module:
    """
    Anchor tiling on elementwise ops only.

    Selects all elementwise linalg ops then performs tile size selection and
    propagation. Already-annotated ops are kept.

    Args:
        tile_size: Tiling size hint.
        strategy: Tiling strategy.
        propagate: Whether to propagate the tile sizes to neighboring ops.
        propagate_through_loops: Whether to propagate through loop-carried
            values instead of treating loops as barriers (default: False).
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(
            named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
        )
        elementwise = transform_ext.filter_elementwise(candidates)
        annotated = transform_ext.assign_tile_sizes(
            elementwise,
            tile_size=tile_size,
            strategy=strategy,
        )
        if propagate:
            transform_ext.propagate_tile_sizes(
                annotated, propagate_through_loops=propagate_through_loops
            )
        transform.yield_()
    return sched


def _execute_annotated(
    target_op: str | list[str] | None,
    use_forall: bool,
    clear_annotations: bool,
    action: str,
) -> ir.Module:
    """Execute annotated tiling actions for matched ops.

    This internal helper applies one of two actions over candidate ops:
    - ``"fuse"``: select fusion roots, tile them using annotated tile sizes,
      and greedily fuse producers.
    - ``"tile"``: tile candidates using annotated tile sizes only.

    Args:
        target_op: Candidate op(s) to consider. Defaults to all linalg ops.
        use_forall: Use ``scf.forall`` for tiling when supported by the action.
        clear_annotations: Clear tile/fuse annotations on produced loop handles.
        action: Action to perform.
    Returns:
        Schedule
    """

    def _merge_loop_handles(loop_handles):
        """Normalize loop handles to a single handle when possible."""
        try:
            loops = list(loop_handles)
        except TypeError:
            return loop_handles
        if len(loops) == 1:
            return loops[0]
        return transform.merge_handles(loops)

    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(named_seq.bodyTarget, target_op)
        if action == "fuse":
            targets = transform_ext.get_fusion_roots(candidates)
        else:
            targets = candidates
        with lh_transform.foreach(targets) as op:
            tiles = transform_ext.get_tile_sizes(op)

            if action == "fuse":
                fused = structured.FuseOp(
                    op,
                    tile_sizes=tiles,
                    apply_cleanup=True,
                    use_forall=use_forall,
                )
                if clear_annotations:
                    loops = _merge_loop_handles(fused.loops)
                    transform_ext.clear_tile_and_fuse_annotations(loops)

            elif action == "tile":
                if use_forall:
                    _, loops = structured.TileUsingForallOp(
                        op,
                        tile_sizes=tiles,
                    ).results
                else:
                    _, loops = structured.TileUsingForOp(
                        op,
                        sizes=tiles,
                    ).results
                if clear_annotations:
                    loops = _merge_loop_handles(loops)
                    transform_ext.clear_tile_and_fuse_annotations(loops)

            else:
                raise ValueError(f"Unsupported action: {action}")

            transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return sched


def tile_and_fuse_annotated(
    target_op: str | list[str] | None = None,
    use_forall: bool = True,
    clear_annotations: bool = True,
) -> ir.Module:
    """
    Tile and fuse annotated groups.

    Fusion roots are selected among the candidates, each is tiled with its
    annotated tile sizes, and its producers are greedily fused into the tiled loop.
    The annotations act as fusion hints.

    Args:
        target_op: Candidate op(s) to consider. Defaults to all linalg ops.
        use_forall: Generate `scf.forall` loops (parallel) when tiling.
        clear_annotations: Clear the annotations from the fused ops.
    Returns:
        Schedule
    """
    return _execute_annotated(
        target_op=target_op,
        use_forall=use_forall,
        clear_annotations=clear_annotations,
        action="fuse",
    )


def tile_annotated(
    target_op: str | list[str] | None = None,
    use_forall: bool = False,
    clear_annotations: bool = True,
) -> ir.Module:
    """
    Tile annotated ops with its tile sizes.

    Args:
        target_op: Candidate op(s) to consider. Defaults to all linalg ops.
        use_forall: Generate `scf.forall` loops (parallel) when tiling.
        clear_annotations: Clear the annotations from the tiled ops.
    Returns:
        Schedule
    """
    return _execute_annotated(
        target_op=target_op,
        use_forall=use_forall,
        clear_annotations=clear_annotations,
        action="tile",
    )


def tile_and_unroll_annotated(
    target_op: str | list[str] | None = None,
    clear_annotations: bool = True,
) -> ir.Module:
    """Tile annotated ops and fully unroll the loops created by tiling."""
    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(named_seq.bodyTarget, target_op)
        with lh_transform.foreach(candidates) as op:
            tiles = transform_ext.get_tile_sizes(op)
            _, loops = structured.TileUsingForOp(
                op,
                sizes=tiles,
            ).results
            if clear_annotations:
                # Clear annotations before unrolling: full unroll may consume
                # loop handles and make post-unroll cleanup invalid.
                transform_ext.clear_tile_and_fuse_annotations(loops)
            inner_to_outer = transform_ext.reverse_handles(loops)
            with lh_transform.foreach(inner_to_outer) as handle:
                loop.loop_unroll_full(handle)
                transform.yield_()
            transform.yield_()
        transform.yield_()
    return sched

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

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
) -> ir.Module:
    """
    Assign tile sizes to anchor ops and propagate them to their neighbors.

    Anchors ops are annotated and their sizes propagated to neighboring ops
    so a fusable group shares one tiling. Can be applied repeatedly with different
    anchors (e.g. GEMMs first, then leftover elementwise ops); already-annotated
    ops keep their sizes, so earlier assignments win.

    Args:
        anchor_op: Op(s) to anchor tiling on. Defaults to the GEMM op family.
        tile_size: Size used for tiled dimensions. User hint, default 32.
    Returns:
        Schedule
    """
    if anchor_op is None:
        anchor_op = _GEMM_ANCHOR_OPS

    with schedule_boilerplate() as (sched, named_seq):
        anchors = lh_transform.match_op(named_seq.bodyTarget, anchor_op)
        annotated = transform_ext.assign_tile_sizes(anchors, tile_size=tile_size)
        transform_ext.propagate_tile_sizes(annotated)
        transform.yield_()
    return sched


def assign_elementwise_tile_sizes(tile_size: int = 32) -> ir.Module:
    """
    Anchor tiling on elementwise ops only.

    Selects all elementwise linalg ops then performs tile size selection and
    propagation. Already-annotated ops are kept.

    Args:
        tile_size: Size used for tiled dimensions. User hint, default 32.
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(
            named_seq.bodyTarget, structured.MatchInterfaceEnum.LinalgOp
        )
        elementwise = transform_ext.filter_elementwise(candidates)
        annotated = transform_ext.assign_tile_sizes(elementwise, tile_size=tile_size)
        transform_ext.propagate_tile_sizes(annotated)
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
    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (sched, named_seq):
        candidates = lh_transform.match_op(named_seq.bodyTarget, target_op)
        roots = transform_ext.get_fusion_roots(candidates)
        with lh_transform.foreach(roots) as op:
            tiles = transform_ext.get_tile_sizes(op)
            fused = structured.FuseOp(
                op,
                tile_sizes=tiles,
                apply_cleanup=True,
                use_forall=use_forall,
            )
            # Fusion has consumed the annotations of this group. Clear them from
            # the ops now inside the generated loop(s).
            if clear_annotations:
                loops = transform.merge_handles(list(fused.loops))
                transform_ext.clear_tile_and_fuse_annotations(loops)
            transform.yield_()
        lh_transform.cleanup(named_seq.bodyTarget)
        transform.yield_()
    return sched

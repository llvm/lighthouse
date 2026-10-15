"""Generate MLIR transform schedule that tiles the payload for WG and reduction dims."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    match,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform import transform_ext
from .attention_fusion import transform_to_flash_attention, annotate_fastmath_flags
from .lowering_common import get_payload_func


def apply_gemm_tiling(
    func: ir.Operation,
    wg_tile: list[int] | None = None,
    k_tile: int | None = None,
    device: str | None = None,
) -> ir.Operation:
    """Apply GEMM tiling to the given function."""

    # Fuse elementwise ops first to simplify the graph.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    matmul_ops = lh_transform.match_op(func, "linalg.matmul")
    # Ensure lowering fails if no matmul ops exist.
    transform_ext.extract_handle(matmul_ops, 0, silenceable=True)
    with lh_transform.foreach(matmul_ops) as matmul_op:
        params = transform_ext.infer_xegpu_gemm_params(
            matmul_op,
            force_wg_tile=wg_tile,
            force_k_tile=k_tile,
            device=device,
        )
        wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
        wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
        _k_tile = transform_ext.get_param_dict_entry(params, "k_tile")
        _wg_tile = [wg_m, wg_n]

        # find the last tileable consumer of the matmul
        consumers = transform_ext.get_tileable_consumers(matmul_op)
        leaf_consumer_op = transform_ext.extract_handle(consumers, -1, silenceable=True)

        # wg tiling
        _, [wg_loop], _ = lh_transform.tile(
            leaf_consumer_op,
            tile_sizes=_wg_tile,
            fuse_producers=True,
            use_forall=True,
            apply_cleanup=False,
        )
        transform.apply_dce(wg_loop)

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        lh_transform.tile(
            wg_matmul,
            tile_sizes=[0, 0, _k_tile],
            fuse_producers=True,
        )
        lh_transform.cleanup(wg_loop)
        transform.yield_()
    lh_transform.cleanup(func)
    return func


def apply_reduction_tiling(func: ir.Operation) -> ir.Operation:
    """Apply reduction tiling to the given function."""

    anytype = transform.AnyOpType.get()

    # Fuse elementwise ops first to simplify the graph.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # TODO implement actual pattern matching and anchor op inference
    anchor_op = lh_transform.match_op(func, "linalg.generic")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    wg_tile, _, reduction_tile = transform_ext.infer_xegpu_reduction_params(anchor_op)

    # WG row tiling
    generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
    leaf_generic = transform_ext.extract_handle(generic_ops, -1, silenceable=True)
    _, [wg_loop], _ = lh_transform.tile(
        leaf_generic,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    def fuse_elemwise_producers_to_loop(target, parent_loop):
        """Fuses all elementwise producer ops of `target` into `parent_loop`."""
        producers = transform_ext.trace_producers(target)
        elemwise_producers = transform_ext.filter_elementwise(producers)
        elemwise_producers = transform_ext.filter_by_name(
            elemwise_producers,
            "linalg.generic",
        )
        _, fused_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=elemwise_producers,
            containing_op=parent_loop,
        )
        return fused_loop

    def tile_and_fuse_reduction(reduction_op, tile_sizes):
        # Tile the reduction op.
        tiled_op, tile_loops, _ = lh_transform.tile(
            reduction_op,
            tile_sizes=tile_sizes,
            fuse_producers=False,
            use_forall=False,
            apply_cleanup=False,
        )
        fuse_elemwise_producers_to_loop(tiled_op, tile_loops[0])

    # Reduction tiling is always applied; the param carries the sizes.
    apply_reduction_tiling = True

    if apply_reduction_tiling:
        # Reduction dimension tiling.
        # 1. Tile the leaf elemwise linalg.generic op and fuse its elemwise
        #    linalg.generic producers into the resulting loop.
        # 2. Tile each reduction linalg.generic op (from last to first) and fuse its
        #    elemwise producers into the resulting loop.

        wg_loop = transform_ext.extract_handle(
            lh_transform.match_op(func, "scf.forall"), 0
        )
        generic_ops = match(wg_loop, ops={"linalg.generic"})
        elemwise_ops = transform_ext.filter_elementwise(generic_ops)
        leaf_elemwise = transform_ext.extract_handle(elemwise_ops, -1, silenceable=True)
        reduction_ops = transform_ext.filter_reduction_ops(generic_ops)

        # Tile trailing elemwise op first.
        res = structured.TileUsingForOp(leaf_elemwise, sizes=reduction_tile).results
        # NOTE tile_using_for can return 2 or 3 values
        tiled_elemwise = res[0]
        tile_loop = res[1]

        # Fuse all elemwise producers into the tiled leaf loop.
        elemwise_for = fuse_elemwise_producers_to_loop(tiled_elemwise, tile_loop)

        # Sink the loop's init tensor.extract_slice into the loop, dropping the
        # surrounding extract/insert_slice pair. This is required for clean
        # vectorization.
        elemwise_for = transform_ext.sink_extract_slice_into_loop(elemwise_for)

        # Tile and fuse the reduction loops in reverse order. After each fusion
        # step, DCE removes the dead untiled elementwise epilogue so it cannot
        # create a cross-loop use that breaks the next tile-fuse iteration. Note
        # that DCE does not invalidate the reduction loop handles as the tracking
        # listener only invalidates modified handles and the reduction loops are
        # alive and thus not removed.
        reduction_ops = transform_ext.reverse_handles(reduction_ops)
        with lh_transform.foreach(reduction_ops) as reduction_op:
            tile_and_fuse_reduction(reduction_op, reduction_tile)
            transform.apply_dce(wg_loop)
            transform.yield_()

        # Fuse all sibling elementwise ops in scf.for loops.
        func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()

    # Cleanup after tiling and fusion.
    lh_transform.cleanup(func)
    return func


def apply_attention_tiling(func: ir.Operation) -> ir.Operation:
    """Apply attention-layer workgroup tiling to the given function."""
    anytype = transform.AnyOpType.get()

    # Clean up to fully expanded shapes by eliminating expand/collapse shape pairs.
    # Generalize all transpose ops.
    transpose_ops = lh_transform.match_op(func, "linalg.transpose")
    structured.structured_generalize(anytype, transpose_ops)
    for i in range(3):
        # Fold reshape ops by expansion and fold extract_slice into tensor.empty
        # Repeat until no more folding.
        func = transform_ext.fold_reshape_ops_by_expansion(func)
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            tensor.apply_patterns_tensor_fold_tensor_empty(fold_single_use_only=False)
    lh_transform.cleanup(func)

    # Find the subtract and exp reduction loops and fuse them.
    subf_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "arith.subf"),
        0,
        silenceable=True,
    )
    subf_loop = transform.get_parent_op(anytype, subf_op, op_name="linalg.generic")
    exp_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "math.exp"),
        0,
        silenceable=True,
    )
    exp_loop = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")
    transform_ext.fuse_elementwise_op(subf_loop, exp_loop)

    lh_transform.cleanup(func)

    # Move the normalizing divide past @V to form the deferred-divide chain.
    contraction_ops = transform_ext.filter_contraction_ops(
        structured.structured_match(anytype, func, ops=["linalg.generic"])
    )
    pv_contraction = transform_ext.extract_handle(contraction_ops, -1, silenceable=True)
    transform_ext.sink_normalization_past_contraction(pv_contraction)
    lh_transform.cleanup(func)

    # Apply WG tiling.
    linalg_ops = lh_transform.match_op(func, ["linalg.generic", "linalg.batch_matmul"])
    leaf_linalg_op = transform_ext.extract_handle(linalg_ops, -1, silenceable=True)
    wg_tile, _sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
        leaf_linalg_op
    )
    lh_transform.tile(
        leaf_linalg_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    func = transform_to_flash_attention(anytype, func, reduction_tile)

    # Inline splat-constant generic inputs (e.g. masked-select zero) as
    # scalars so they are not vectorized into reads of constant tensors.
    func = transform_ext.fold_scalar_or_splat_constant(func)

    annotate_fastmath_flags(func)
    lh_transform.cleanup(func)

    return func


def wg_tiling_schedule(
    wg_tile: list[int] | None = None,
    k_tile: int | None = None,
    payload_func_name: str | None = None,
) -> ir.Module:
    """Tile the payload for workgroup parallelism and the reduction dimension."""

    with schedule_boilerplate() as (schedule, named_seq):
        op_names = ["linalg.generic", "linalg.matmul"]
        payload_func = get_payload_func(
            named_seq.bodyTarget,
            op_name=op_names,
            func_name=payload_func_name,
        )

        # Use alternatives op to try different lowerings. The defined regions
        # are applied in order, if one fails with a SilenceableFailure, the
        # next alternative is tried. The last region emits a definite failure
        # in case none of the patterns match.
        alt = lh_transform.alternatives(
            payload_func, num_alternatives=4, result_types=[payload_func.type]
        )
        with alt.region(0) as func:
            func = apply_gemm_tiling(func, wg_tile=wg_tile, k_tile=k_tile)
            transform.yield_([func])
        with alt.region(1) as func:
            # TODO add support for forcing custom tile sizes
            func = apply_attention_tiling(func)
            transform.yield_([func])
        with alt.region(2) as func:
            # TODO add support for forcing custom tile sizes
            func = apply_reduction_tiling(func)
            transform.yield_([func])
        with alt.region(3) as func:
            transform_ext.emit_definite_failure(
                func, message="wg_tiling: Could not apply any of the defined patterns."
            )
            transform.yield_([func])
        payload_func = alt.results[0]

        transform.yield_()

    return schedule

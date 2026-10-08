"""Generate MLIR transform schedule that tiles the payload for WG and reduction dims."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    match,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform import transform_ext
from .lowering_common import get_payload_func


def apply_gemm_tiling(
    func: ir.Operation,
    wg_tile: list[int] | None = None,
    k_tile: int | None = None,
    device: str | None = None,
) -> ir.Operation:
    """Apply GEMM tiling to the given function."""

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

    # Payloads write the softmax in the conventional order, with the normalizing
    # divide before the `@V` contraction, and the pass above fuse that divide into
    # the contraction's body. Move it past the contraction, so the op sequence is
    # amenable to flash attention style fusion.
    contraction_ops = transform_ext.filter_contraction_ops(
        lh_transform.match_op(func, "linalg.generic")
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
    _, [forall_loop], _ = lh_transform.tile(
        leaf_linalg_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    # Apply reduction tiling and fusion, still at tensor level. The Q, K, V
    # tensors and the scale constant are found by walking the SSA chain of the
    # two batch matmuls inside the WG forall:
    #
    #   Q@K^T:  linalg.batch_matmul(q_slice, linalg.transpose(k_slice))
    #   scale:  linalg.elementwise <mul>(qkt, linalg.fill(scale_constant))
    #   P@V:    linalg.batch_matmul(softmax_out, v_slice)

    linalg_ops = lh_transform.match_op(
        forall_loop, ["linalg.generic", "linalg.batch_matmul"]
    )
    contraction_ops = transform_ext.filter_contraction_ops(linalg_ops)

    # Match max reduction op. Assumes there's only one arith.max* op.
    arith_max_op = transform_ext.extract_handle(
        lh_transform.match_op(forall_loop, ["arith.maximumf", "arith.maxnumf"]),
        0,
        silenceable=True,
    )
    max_reduction = transform.get_parent_op(
        anytype, arith_max_op, op_name="linalg.generic"
    )

    def get_producers_by_name(target, op_names):
        producers = transform_ext.trace_producers(target)
        return transform_ext.filter_by_name(producers, op_names=op_names)

    # Trace the scalar from max reduction producer chain.
    max_producer_generics = get_producers_by_name(
        max_reduction, op_names="linalg.generic"
    )
    # Assume the scaling happens in the first ancestor.
    max_producer = transform_ext.extract_handle(
        max_producer_generics, 0, silenceable=True
    )
    max_scale_mul_op = transform_ext.extract_handle(
        lh_transform.match_op(max_producer, "arith.mulf"), 0, silenceable=True
    )
    scale_producers = transform_ext.trace_producers(max_scale_mul_op)
    scale_const_op = transform_ext.extract_handle(
        transform_ext.filter_by_name(scale_producers, op_names="arith.constant"),
        0,
        silenceable=True,
    )

    matmul_ops = transform.split_handle(2 * [anytype], contraction_ops)
    qk_matmul, pv_matmul = matmul_ops[0], matmul_ops[1]

    # Q is the Q@K^T contraction's first operand. K is the second operand
    # before the transpose.
    q = transform.get_producer_of_operand(anytype, qk_matmul, 0)
    k_producer = transform.get_producer_of_operand(anytype, qk_matmul, 1)
    all_k_producers = transform_ext.trace_producers(k_producer)
    all_k_producers = transform.merge_handles([k_producer, all_k_producers])
    k_transpose = transform_ext.filter_by_name(
        all_k_producers, op_names="linalg.transpose"
    )
    k_transpose = transform_ext.extract_handle(k_transpose, 0, silenceable=True)
    k = transform.get_producer_of_operand(anytype, k_transpose, 0)

    # Find handle to v as the first tensor.extract_slice producer of PV matmul
    pv_extract_slice_producers = get_producers_by_name(
        pv_matmul, op_names="tensor.extract_slice"
    )
    v = transform_ext.extract_handle(pv_extract_slice_producers, 0)

    # Replace the region's result with a loop over the K/V sequence length that
    # implements online softmax, fusing Q@K^T and the softmax into it. The emitted
    # loop ends in `acc / l`, so what it replaces is the normalizing divide the sink
    # above moved past the contraction -- the contraction's only consumer. The
    # contraction and the rest of the softmax chain are then dead and DCE'd. (Once
    # the reduction fusion replaces this op, the sunk chain is folded into a loop
    # directly instead of being rebuilt from q/k/v.)
    normalize_op = transform.get_consumers_of_result(anytype, pv_matmul, 0)
    # Match P by its math.exp op. PV matmul can have a mask as the 1st operand.
    exp_op = transform_ext.extract_handle(
        lh_transform.match_op(forall_loop, "math.exp"), 0, silenceable=True
    )
    p = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")
    transform_ext.replace_with_fused_attention(
        q=q,
        k=k,
        v=v,
        p=p,
        scale=scale_const_op,
        replaced=normalize_op,
        tile_size=reduction_tile,
    )
    transform.apply_cse(func)
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

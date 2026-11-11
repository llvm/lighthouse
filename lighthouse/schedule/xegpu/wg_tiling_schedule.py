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
from .attention_fusion import transform_to_flash_attention
from .lowering_common import get_payload_func
from lighthouse.dialects.transform.transform_ext.utils.subgraph_analysis import (
    SUBGRAPH_ANCHOR_ATTR_NAME,
    SUBGRAPH_ID_ATTR_NAME,
    SUBGRAPH_KIND_ATTR_NAME,
)


def _subgraph_anchors(func: ir.Value, kind: str) -> ir.Value:
    """Match the anchors of the subgraphs of `kind` tagged by `tag_subgraphs`."""
    return match(
        func,
        op_attrs={
            SUBGRAPH_KIND_ATTR_NAME: ir.StringAttr.get(kind),
            SUBGRAPH_ANCHOR_ATTR_NAME: ir.UnitAttr.get(),
        },
    )


def _tile_subgraph(
    anchor: ir.Value,
    producers: ir.Value,
    consumers: ir.Value,
    wg_tile: list | ir.Value,
) -> ir.Value:
    """Tile the anchor into an scf.forall and fuse its subgraph into it.

    Both producers and consumers must be supplied in program order, as returned
    by get_subgraph_ops. This helper reverses producers before fusing them so
    their consumers are fused first, including all users of shared producers.
    Consumers of the anchor are then fused forward in the supplied order.
    Returns the workgroup loop.
    """
    anytype = transform.AnyOpType.get()
    _, wg_loop = structured.TileUsingForallOp(anchor, tile_sizes=wg_tile).results
    _, wg_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=transform_ext.reverse_handles(producers),
        containing_op=wg_loop,
    )
    return transform_ext.fuse_forall_epilogue(wg_loop, consumers)


def apply_subgraph_tiling(
    func: ir.Operation,
    force_wg_tile: list[int] | None = None,
    force_k_tile: int | None = None,
    device: str | None = None,
) -> ir.Operation:
    """Partition attention, reduction, GEMM and elementwise subgraphs and tile
    each into its own workgroup-parallel scf.forall, i.e. its own kernel.

    Reshape preparation, cast normalization and shared-producer isolation run
    before annotation-only tagging or tiling.
    Attention is claimed first so softmax stays with its contractions.
    Normalization reductions are claimed before GEMM to materialize their
    outputs once, including when shared by multiple projections.
    GEMM subgraphs then claim their epilogue chains; the
    remaining elementwise ops form separate subgraphs. Pure elementwise graphs
    are also supported. Fails silenceably if a linalg op is left unpartitioned.
    """

    func = _prepare_iteration_spaces(func)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_swap_extract_slice_with_fill()
    kinds = ["attention", "reduction", "gemm", "elementwise"]
    func = transform_ext.prepare_subgraphs(func, kinds)
    transform_ext.tag_subgraphs(func, kinds)

    with lh_transform.foreach(_subgraph_anchors(func, "reduction")) as anchor:
        # Standalone reductions keep the existing schedule and its overrides.
        transform_ext.extract_handle(
            lh_transform.match_op(func, "linalg.matmul"), 0, silenceable=True
        )
        producers, _ = transform_ext.get_subgraph_ops(anchor)
        reduction = transform_ext.filter_reduction_ops(producers)
        wg_tile, _, reduction_tile = transform_ext.infer_xegpu_reduction_params(
            reduction
        )
        id = transform_ext.get_named_attribute(anchor, SUBGRAPH_ID_ATTR_NAME)
        producers = transform_ext.filter_by_name(
            transform_ext.trace_producers(
                anchor, scope=func, subgraph=id, order="reverse-topological"
            ),
            ["linalg.generic", "linalg.fill", "linalg.transpose"],
        )
        _, wg_loop = structured.TileUsingForallOp(anchor, tile_sizes=wg_tile).results
        _, wg_loop = structured.structured_fuse_into_containing_op(
            transform.AnyOpType.get(),
            transform.AnyOpType.get(),
            producer_op=producers,
            containing_op=wg_loop,
        )
        lh_transform.cleanup(wg_loop)
        wg_loop = transform_ext.fuse_elementwise_region(wg_loop)
        lh_transform.cleanup(wg_loop)
        _tile_reduction_body(wg_loop, reduction_tile)
        transform.yield_()
    transform.apply_dce(func)

    with lh_transform.foreach(_subgraph_anchors(func, "gemm")) as anchor:
        producers, consumers = transform_ext.get_subgraph_ops(anchor)
        matmul_op = transform_ext.filter_by_name(
            transform.merge_handles([anchor, producers]), "linalg.matmul"
        )
        params = transform_ext.infer_xegpu_gemm_params(
            matmul_op,
            force_wg_tile=force_wg_tile,
            force_k_tile=force_k_tile,
            device=device,
        )
        wg_m = transform_ext.get_param_dict_entry(params, "wg_m")
        wg_n = transform_ext.get_param_dict_entry(params, "wg_n")
        k_tile = transform_ext.get_param_dict_entry(params, "k_tile")

        wg_loop = _tile_subgraph(anchor, producers, consumers, [wg_m, wg_n])

        # k loop tiling
        wg_matmul = match(wg_loop, ops={"linalg.matmul"})
        lh_transform.tile(
            wg_matmul,
            tile_sizes=[0, 0, k_tile],
            fuse_producers=True,
        )
        lh_transform.cleanup(wg_loop)
        transform.yield_()
    transform.apply_dce(func)

    with lh_transform.foreach(_subgraph_anchors(func, "elementwise")) as anchor:
        producers, consumers = transform_ext.get_subgraph_ops(anchor)
        wg_tile, _ = transform_ext.infer_xegpu_elementwise_params(anchor)
        wg_loop = _tile_subgraph(anchor, producers, consumers, wg_tile)
        lh_transform.cleanup(wg_loop)
        transform.yield_()

    attention_ids = transform_ext.get_named_attribute(
        _subgraph_anchors(func, "attention"), SUBGRAPH_ID_ATTR_NAME
    )
    with lh_transform.foreach(attention_ids):
        # Standalone attention retains its schedule overrides and fallback path.
        transform_ext.extract_handle(
            lh_transform.match_op(func, "linalg.matmul"), 0, silenceable=True
        )
        transform.yield_()
    with lh_transform.foreach(attention_ids) as id:
        apply_attention_tiling(func, subgraph=id)
        transform.yield_()
    func = apply_registered_pass(func, "remove-dead-values")

    transform_ext.clear_subgraph_tags(func)
    func = transform_ext.split_concat_output(func)
    lh_transform.cleanup(func)
    return func


def _tile_reduction_body(wg_loop, reduction_tile):
    """Tile a normalization kernel's output and reductions within its WG."""
    anytype = transform.AnyOpType.get()

    def fuse_producers(target, loop):
        producers = transform_ext.filter_by_name(
            transform_ext.filter_elementwise(
                transform_ext.trace_producers(target, scope=wg_loop)
            ),
            "linalg.generic",
        )
        _, loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=producers, containing_op=loop
        )
        return loop

    generic_ops = match(wg_loop, ops={"linalg.generic"})
    leaf = transform_ext.extract_handle(
        transform_ext.filter_elementwise(generic_ops), -1, silenceable=True
    )
    reductions = transform_ext.reverse_handles(
        transform_ext.filter_reduction_ops(generic_ops)
    )
    res = structured.TileUsingForOp(leaf, sizes=reduction_tile).results
    loop = fuse_producers(res[0], res[1])
    transform_ext.sink_extract_slice_into_loop(loop)
    with lh_transform.foreach(reductions) as reduction:
        tiled, loops, _ = lh_transform.tile(
            reduction,
            tile_sizes=reduction_tile,
            fuse_producers=False,
            use_forall=False,
            apply_cleanup=False,
        )
        fuse_producers(tiled, loops[0])
        transform.apply_dce(wg_loop)
        transform.yield_()


def apply_reduction_tiling(
    func: ir.Operation,
    force_wg_tile: list[int] | None = None,
    force_wg_subtile: list[int] | None = None,
    force_reduction_tile: list[int] | None = None,
) -> ir.Operation:
    """Apply reduction tiling to the given function."""

    anytype = transform.AnyOpType.get()

    # Fuse elementwise ops first to simplify the graph.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # TODO implement actual pattern matching and anchor op inference
    anchor_op = lh_transform.match_op(func, "linalg.generic")
    anchor_op = transform_ext.extract_handle(anchor_op, 0, silenceable=True)
    wg_tile, _, reduction_tile = transform_ext.infer_xegpu_reduction_params(
        anchor_op,
        force_wg_tile=force_wg_tile,
        force_reduction_tile=force_reduction_tile,
    )

    # Adding persistent reduction loop is only possible via forced args for now.
    wg_subtile = None
    if force_wg_subtile is not None:
        wg_subtile = force_wg_subtile
    apply_wg_subtile = wg_subtile is not None and sum(wg_subtile) > 0

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

    if apply_wg_subtile:
        # Add a persistent loop to reduce wg tile size.
        generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
        leaf_generic = transform_ext.extract_handle(generic_ops, -1)
        lh_transform.tile(
            leaf_generic,
            tile_sizes=wg_subtile,
            fuse_producers=True,
            use_forall=False,
            apply_cleanup=False,
        )
        lh_transform.cleanup(func)

    wg_loop = transform_ext.extract_handle(lh_transform.match_op(func, "scf.forall"), 0)
    _tile_reduction_body(wg_loop, reduction_tile)
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()

    # Cleanup after tiling and fusion.
    lh_transform.cleanup(func)
    return func


def _prepare_iteration_spaces(func):
    """Prepare generic iteration spaces before subgraph tagging and WG tiling."""
    for i in range(3):
        # Fold reshape ops by expansion and fold extract_slice into tensor.empty
        # Repeat until no more folding.
        func = transform_ext.fold_reshape_ops_by_expansion(func)
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            tensor.apply_patterns_tensor_fold_tensor_empty(fold_single_use_only=False)
    lh_transform.cleanup(func)
    return func


def apply_attention_tiling(
    func: ir.Operation,
    force_wg_tile: int | None = None,
    force_reduction_tile: int | None = None,
    subgraph: ir.Value | None = None,
) -> ir.Operation:
    """Tile attention, including causal score producers, into one WG kernel.

    With a subgraph id, shapes must already be prepared. Matching and initial
    fusion are restricted to its members; online fusion stays inside the WG.
    No full score or mask tensor is stored, and mask indices retain global
    query/key offsets.
    """
    anytype = transform.AnyOpType.get()
    if subgraph is None:
        func = _prepare_iteration_spaces(func)

    def attention_ops(names):
        if subgraph is None:
            return lh_transform.match_op(func, names)
        return transform_ext.match_subgraph(func, subgraph, names, role="body")

    # Find the subtract and exp reduction loops and fuse them.
    subf_op = transform_ext.extract_handle(
        attention_ops("arith.subf"),
        0,
        silenceable=True,
    )
    subf_loop = transform.get_parent_op(anytype, subf_op, op_name="linalg.generic")
    exp_op = transform_ext.extract_handle(
        attention_ops("math.exp"),
        0,
        silenceable=True,
    )
    exp_loop = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")
    fused = transform_ext.fuse_elementwise_op(subf_loop, exp_loop)
    if subgraph is not None:
        transform.annotate(fused, SUBGRAPH_ID_ATTR_NAME, param=subgraph)

    lh_transform.cleanup(func)

    # Move the normalizing divide past @V to form the deferred-divide chain.
    contraction_ops = transform_ext.filter_contraction_ops(
        attention_ops("linalg.generic")
    )
    pv_contraction = transform_ext.extract_handle(contraction_ops, -1, silenceable=True)
    contraction, normalization = transform_ext.sink_normalization_past_contraction(
        pv_contraction
    )
    if subgraph is not None:
        transform.annotate(contraction, SUBGRAPH_ID_ATTR_NAME, param=subgraph)
        transform.annotate(normalization, SUBGRAPH_ID_ATTR_NAME, param=subgraph)
    lh_transform.cleanup(func)

    # Move the uniform softmax scale from K onto Q: Q is loaded once per WG,
    # whereas K is reloaded in every reduction iteration.
    qk_contraction = transform_ext.extract_handle(
        transform_ext.filter_contraction_ops(attention_ops("linalg.generic")),
        0,
        silenceable=True,
    )
    transform_ext.move_contraction_scale_to_lhs(qk_contraction)
    lh_transform.cleanup(func)

    # Apply WG tiling.
    linalg_ops = attention_ops(["linalg.generic", "linalg.batch_matmul"])
    leaf_linalg_op = transform_ext.extract_handle(linalg_ops, -1, silenceable=True)
    wg_tile, _sg_tile, reduction_tile = transform_ext.infer_xegpu_attention_params(
        leaf_linalg_op,
        force_wg_tile=force_wg_tile,
        force_reduction_tile=force_reduction_tile,
    )
    if subgraph is None:
        _, [wg_loop], _ = lh_transform.tile(
            leaf_linalg_op,
            tile_sizes=wg_tile,
            fuse_producers=True,
            use_forall=True,
            apply_cleanup=False,
        )
    else:
        # A shared producer must follow every consumer, not merely its nearest one.
        producers = transform_ext.filter_by_name(
            transform_ext.trace_producers(
                leaf_linalg_op,
                scope=func,
                subgraph=subgraph,
                order="reverse-topological",
            ),
            [
                "linalg.generic",
                "linalg.batch_matmul",
                "linalg.transpose",
                "linalg.fill",
            ],
        )
        _, wg_loop = structured.TileUsingForallOp(
            leaf_linalg_op, tile_sizes=wg_tile
        ).results
        _, wg_loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=producers, containing_op=wg_loop
        )
    lh_transform.cleanup(func)

    wg_loop = transform_to_flash_attention(anytype, wg_loop, reduction_tile)
    if subgraph is not None:
        consumers = transform_ext.match_subgraph(func, subgraph, role="epilogue")
        wg_loop = transform_ext.fuse_forall_epilogue(wg_loop, consumers)
    if subgraph is None:
        func = apply_registered_pass(func, "remove-dead-values")

    # Inline splat-constant generic inputs (e.g. masked-select zero) as
    # scalars so they are not vectorized into reads of constant tensors.
    if subgraph is None:
        func = transform_ext.fold_scalar_or_splat_constant(func)
    else:
        transform_ext.fold_scalar_or_splat_constant(wg_loop)
    lh_transform.cleanup(func)

    # Write the result through an expanded view of the output memref so the GQA
    # collapse_shape leaves the write path and bufferization elides the output
    # buffer + copy.
    func = transform_ext.expand_output_destination(func)
    lh_transform.cleanup(func)

    return func


def wg_tiling_schedule(
    wg_tile: list[int] | None = None,
    k_tile: int | None = None,
    reduction_tile: list[int] | None = None,
    payload_func_name: str | None = None,
    wg_subtile: list[int] | None = None,
    device: str | None = None,
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
            func = apply_subgraph_tiling(
                func,
                force_wg_tile=wg_tile,
                force_k_tile=k_tile,
                device=device,
            )
            transform.yield_([func])
        with alt.region(1) as func:
            # TODO legacy path, remove. apply_subgraph_tiling should be used instead.
            func = apply_attention_tiling(
                func, force_wg_tile=wg_tile, force_reduction_tile=reduction_tile
            )
            transform.yield_([func])
        with alt.region(2) as func:
            # TODO legacy path, remove. apply_subgraph_tiling should be used instead.
            func = apply_reduction_tiling(
                func,
                force_wg_tile=wg_tile,
                force_wg_subtile=wg_subtile,
                force_reduction_tile=reduction_tile,
            )
            transform.yield_([func])
        with alt.region(3) as func:
            transform_ext.emit_definite_failure(
                func, message="wg_tiling: Could not apply any of the defined patterns."
            )
            transform.yield_([func])
        payload_func = alt.results[0]
        payload_func = transform_ext.expand_output_destination(payload_func)
        lh_transform.cleanup(payload_func)

        # Fold unit dimensions.
        # Generalize named ops so unit dims in transpose ops are folded too.
        structured.structured_generalize(
            transform.AnyOpType.get(),
            lh_transform.match_op(payload_func, "linalg.transpose").results[0],
        )
        with ir.InsertionPoint(transform.apply_patterns(payload_func).patterns):
            structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
            tensor.apply_patterns_tensor_reassociative_reshape_folding()
        lh_transform.cleanup(payload_func)

        transform.yield_()

    return schedule

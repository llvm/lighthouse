"""Generate MLIR transform schedule for XeGPU fused attention layer."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, xegpu, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from .lowering_common import (
    get_payload_func,
    vectorize,
    bufferize,
    convert_to_gpu_launch,
    convert_vector_to_xegpu,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.dialects.transform import transform_ext


def fused_attention_schedule(
    stop_at_stage: str | None = None,
    params: ScheduleParameters | None = None,
) -> ir.Module:
    """
    Generate transform schedule for attention kernel.

    Input matrices are Q, K, V. All inputs, as well as the layer output, have
    the same shape [batch_size, n_head, n_ctx, d_head] where

        - batch_size: number of sequences in the batch
        - n_head: number of attention heads
        - n_ctx: sequence length (number of tokens)
        - d_head: dimension of each attention head

    The attention layer computes the following operations:

        1. S = Q @ K^T - batch matmul, contraction over last d_head dim of Q
            S has shape [batch_size, n_head, n_ctx, n_ctx]
        2. P = Softmax(S, axis=-1)  - softmax over the last dimension
            P has shape [batch_size, n_head, n_ctx, n_ctx]
        3. O = P @ V - batch matmul, contraction over the last n_ctx dim of P

    The schedule performs the following transformations:

        1. Tile and fuse the attention computation along parallel dims
        2. Perform the fused attention optimization for the innermost block
        3. Vectorize operations
        4. Bufferize tensors
        5. Convert to GPU dialect
        6. Lower to XeGPU operations

    Step 1. applies parallel tiling for the workgroup and subgroup levels
    over the first (batch_size, n_head, n_ctx) dimensions, controlled by the
    `wg_tile` and `sg_rows` parameters.

    If the linalg ops operate on the 4d input tensors, the `wg_tile` parameter
    should take the form [1, 1, wg_rows] where wg_rows denotes the workgroup
    n_ctx tile size. If the batch_size and n_head dimensions have been
    collapsed into a single batch dimension, the `wg_tile` parameter should
    take the form [1, wg_rows].

    The `sg_rows` parameter controls the number of rows per subgroup, applied
    over the n_ctx dimension.

    Given the `wg_tile` and `sg_rows` parameters, the WG tile is expected to be
    of form [1, ..., wg_rows], depending on the number of leading parallel
    dimensions, and the `sg_rows` tiling is applied over the n_ctx dimension.

    In step 2., the inner attention block is folded into a single loop over the
    key/value axis, controlled by the `reduction_tile` parameter: the row max is
    tiled along that axis and the rest of the chain -- the exp term, the row sum,
    the P@V contraction and the score computation -- is fused into it, giving the
    online (flash) softmax. This happens at tensor level, so the tiling and fusion
    decisions stay at the level the rest of the schedule works on and the emitted
    loop is lowered by the regular vectorization step.

    Prefetching of K and V tiles is controlled by the `prefetch_tile` and
    `nb_prefetch` parameters.

    Expects a `ScheduleParameters` object with a single dictionary containing
    the following keys:

        - layer_kind: "attention"
        - batch_size: int
        - n_head: int
        - n_ctx: int
        - d_head: int
        - wg_tile: list[int, ...]
        - sg_rows: int, applied on n_ctx dim
        - reduction_tile: int, applied on last n_ctx dim of P
        - subgroup_size: int
        - q_load_tile: list[int, int]
        - v_load_tile: list[int, int]
        - prefetch_tile: list[int, int]
        - nb_prefetch: int

    Returns:
        MLIR module containing the transform schedule
    """
    assert params is not None and len(params) > 0, (
        "Schedule parameters must be provided"
    )

    with schedule_boilerplate() as (schedule, named_seq):
        # match the payload module
        anytype = transform.AnyOpType.get()
        func = match(named_seq.bodyTarget, ops={"func.func"})
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )

        try:
            bundle_xegpu_fused_attention_schedule(
                payload_mod,
                params=params,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def _derive_flash_attention(anytype, func, layer_params):
    """Derive the flash loop from the payload chain with the reduction fusion.

    `fuse_dependant_reduction_ops` moves the elementwise term and one consumer
    reduction into an already-tiled producer reduction loop and inserts the online
    correction that rescales that reduction's running accumulator whenever the
    running max changes. Applied once per consumer reduction -- the row sum and the
    `P@V` contraction -- it folds the whole chain into a single loop.

    Inside the WG forall the chain reads, in program order:

        %s   = linalg.mul(batch_matmul(q, k^T), fill(scale))   the scaled scores
        %m   = max_k %s                                        the producer reduction
        %p   = exp(%s - %m)                                    the elementwise term
        %o   = contract(%p, v)                                 consumer reduction
        %l   = sum_k %p                                        consumer reduction
        %out = %o / %l                                         the deferred divide
    """
    # Tile size for the reduction dimension (the K/V sequence length).
    reduction_tile = layer_params["reduction_tile"]

    # Program order at this point (the WG tiling reorders the two consumer
    # reductions relative to the payload's build order): row max, exp term, the P@V
    # contraction, row sum, deferred divide. P@V is already a `linalg.generic` --
    # the payload writes it that way so the narrowing of P can sit in its body (see
    # `generate_gpu_attention_payload`) -- so unlike Q@K^T it needs no `generalize`
    # to be a fusable consumer.
    max_op, p_op, pv_op, sum_op, _divide = match_and_split(
        func, ops={"linalg.generic"}, nhandles=5
    )

    # Tile the row max along the key/value axis. This is the producer reduction
    # loop the rest of the chain gets folded into; the marker attribute is what
    # the fusion op recognizes it by.
    _, reduction_loop = structured.structured_tile_using_for(
        anytype,
        [anytype],
        max_op,
        dynamic_sizes=[],
        interchange=[],
        static_sizes=[0, 0, reduction_tile],
        scalable_sizes=[False, False, False],
    )
    transform.annotate(reduction_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

    # First chain: max -> p -> row sum. `p` also feeds the contraction, so the op
    # fuses a clone of it and leaves the original in place for the second chain.
    # Fusing replaces the loop, but the replacement inherits the marker attribute,
    # so it is ready to serve as the producer reduction of the second chain.
    reduction_loop = transform_ext.fuse_dependant_reduction_ops(
        p_op, sum_op, reduction_loop
    )

    # Second chain: max -> p -> P@V, into that same loop. The first fusion
    # consumed the handle to `p`; the original is still the contraction's operand.
    p_op = transform.get_producer_of_operand(anytype, pv_op, operand_number=0)
    reduction_loop = transform_ext.fuse_dependant_reduction_ops(
        p_op, pv_op, reduction_loop
    )
    transform.apply_cse(func)

    # Sink the score computation into the reduction loop as well, so only one
    # [wg_rows, tile_size] score tile -- rather than the full [wg_rows, n_ctx]
    # matrix -- is ever live.
    for producer_name in ["linalg.mul", "linalg.batch_matmul", "linalg.transpose"]:
        producer_op = match_and_split(func, ops={producer_name}, nhandles=1)[0]
        _, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=producer_op,
            containing_op=reduction_loop,
        )

    # The Q @ K^T zero accumulator and the scale tensor are still filled at full
    # [wg_rows, n_ctx] extent outside the loop, even though the ops now inside it
    # only ever read a [wg_rows, tile_size] slice. Sink those two fills as well so
    # neither tensor is materialized whole. Reach them through the in-loop consumer
    # they initialize -- one hop for the slice the fusion left behind, one more for
    # the fill itself. The three remaining fills initialize the loop's running
    # accumulators and must stay outside.
    scale_mul_op = match_and_split(reduction_loop, ops={"linalg.mul"}, nhandles=1)[0]
    qk_matmul = match_and_split(
        reduction_loop, ops={"linalg.batch_matmul"}, nhandles=1
    )[0]
    for consumer_op, operand_number in [(scale_mul_op, 1), (qk_matmul, 2)]:
        fill_slice = transform.get_producer_of_operand(
            anytype, consumer_op, operand_number=operand_number
        )
        fill_op = transform.get_producer_of_operand(
            anytype, fill_slice, operand_number=0
        )
        _, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=fill_op,
            containing_op=reduction_loop,
        )

    transform.apply_cse(func)
    canonicalize(func)

    # Strip the batch dim, which the work-group tiling cut down to 1. Everything
    # downstream -- the XeGPU layouts below and the blocking/distribution passes in
    # `xegpu_to_binary` -- is built for rank-2 tiles, and the vector-level
    # `cast_away_leading_one_dim` patterns cannot finish the job: they have no
    # pattern for multi_reduction, broadcast or transpose, so the softmax row
    # reductions and the correction's broadcasts would keep a unit dim and drag
    # shape_casts (and rank-3 XeGPU layouts) along with them.
    #
    # Done here, after the reduction fusion rather than before it, for two reasons:
    # `fuse_dependant_reduction_ops` gets to run on the shape it already handles,
    # and every op above is still matched by name -- generalizing first would turn
    # the whole chain into linalg.generic.
    #
    # `fold_unit_extent_dims` only rewrites linalg.generic, hence the generalize;
    # and it leaves two-step slice chains behind (16x4096x64 -> 1x128x64 ->
    # 128x64), which plain canonicalization does not compose, hence the tensor
    # patterns. The scf.for's own iter_args keep their unit dim -- no pattern
    # retypes a loop signature -- but that no longer matters: with every op inside
    # rank 2, vectorization reads through them rank-reducing and the accumulators
    # come out as rank-1/2 vectors.
    named_ops = match(
        func,
        ops={
            "linalg.batch_matmul",
            "linalg.mul",
            "linalg.transpose",
            "linalg.fill",
            "linalg.elementwise",
        },
    )
    structured.structured_generalize(anytype, named_ops)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        tensor.apply_patterns_tensor_merge_consecutive_insert_extract_slice()
        tensor.apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion()
        tensor.apply_patterns_tensor_fold_tensor_subset_ops()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)


def _replace_with_reference_flash_attention(anytype, func, layer_params):
    """Emit the flash loop from the hand-written generator instead of deriving it.

    `replace_with_fused_attention` builds the whole online-softmax loop from
    scratch given Q, K, V and the scale, replacing the chain's leaf. It is kept as
    a reference point: the derived path (the default, see
    `fuse_dependant_reduction_ops`) should converge on the same loop, and diffing
    the two at `--dump-kernel=reduction-tiled` is how that is tracked.

    Both paths consume the same payload. The generator's own output *is* the
    normalized result, so its `output` is the payload's deferred divide -- the
    chain's leaf -- rather than the `P@V` contraction; everything upstream of it
    (max, exp, row sum, `P@V`) is left dead for DCE.
    """
    prod = transform.get_producer_of_operand
    # The Q, K, V tensors and the scale constant are found by walking the SSA chain
    # of the two batch matmuls inside the WG forall:
    #
    #   Q@K^T:  linalg.batch_matmul(q_slice, linalg.transpose(k_slice))
    #   scale:  linalg.mul(qkt, linalg.fill(scale_constant))
    #   P@V:    linalg.generic(probs, v_slice)   -- a generic, not a named matmul,
    #                                              so that P's narrowing can live in
    #                                              its body; V is its rhs either way
    qk_matmul = match_and_split(func, ops={"linalg.batch_matmul"}, nhandles=1)[0]
    q = prod(anytype, qk_matmul, operand_number=0)
    k_transpose = prod(anytype, qk_matmul, operand_number=1)
    k = prod(anytype, k_transpose, operand_number=0)
    mul_op = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]
    scale = prod(anytype, prod(anytype, mul_op, operand_number=1), operand_number=0)
    # Generics in program order: row max, exp term, P@V, row sum, deferred divide.
    # The divide is the chain's leaf; V is P@V's rhs.
    _max, _exp, pv_op, _sum, divide_op = match_and_split(
        func, ops={"linalg.generic"}, nhandles=5
    )
    v = prod(anytype, pv_op, operand_number=1)

    transform_ext.replace_with_fused_attention(
        q=q,
        k=k,
        v=v,
        scale=scale,
        output=divide_op,
        tile_size=layer_params["reduction_tile"],
    )
    transform.apply_cse(func)
    lh_transform.cleanup(func)


def bundle_xegpu_fused_attention_schedule(
    mod: ir.Value[transform.AnyOpType],
    params: ScheduleParameters,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering attention payload to xegpu wg level."""

    layer_params = params[0]

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # Match payload function
    func = get_payload_func(mod, op_name=["linalg.generic", "linalg.batch_matmul"])

    # The payload spells the softmax out as `max -> exp -> {sum, P@V} -> divide`
    # (see `generate_gpu_attention_payload`), so there is no `linalg.softmax` to
    # decompose: its decomposition normalizes *before* the contraction, which
    # would leave the P@V reading the normalized P and so break the dependency
    # chain the reduction fusion needs.

    # Normalize possible singleton dimensions so tile+fuse logic works.
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        # fold unit dims in linalg.generic op inputs
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        # fold tensor.extract_slice(tensor.expand_shape(x)) into x
        tensor.apply_patterns_tensor_reassociative_reshape_folding()
        # swap tensor.extract_slice(linalg.fill(...)) ops
        structured.apply_patterns_linalg_swap_extract_slice_with_fill()
        # fold tensor.extract_slice(tensor.empty(...)) into tensor.tensor_empty(...)
        tensor.apply_patterns_tensor_fold_tensor_empty(fold_single_use_only=True)
    lh_transform.cleanup(func)

    # Fuse elementwise ops, also removes unused linalg op results (if any).
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")
    lh_transform.cleanup(func)

    # Apply WG tiling
    wg_tile = layer_params["wg_tile"]
    for wg in wg_tile[:-1]:
        assert wg == 1, "WG tile must be of form [1, ..., wg_rows]"
    wg_rows = wg_tile[-1]

    linalg_ops = structured.structured_match(
        anytype, func, ops=["linalg.generic", "linalg.batch_matmul"]
    )
    leaf_linalg_op = transform_ext.extract_handle(linalg_ops, -1)
    leaf_generic_wg, _, _ = lh_transform.tile(
        leaf_linalg_op,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # Build the fused (flash) attention inner loop -- a single loop over the
    # key/value axis -- while still at linalg level. Two paths produce it, and both
    # consume the same payload, so their output can be diffed at
    # `--dump-kernel=reduction-tiled`:
    #
    #   * the default derives it from the payload's chain with
    #     `fuse_dependant_reduction_ops` (below);
    #   * `reference_flash` emits it from the hand-written generator instead, as a
    #     reference point for how close the derived version gets.
    # Tile size for the reduction dimension (the K/V sequence length); also drives
    # the K/V prefetch and the XeGPU layouts further down.
    reduction_tile = layer_params["reduction_tile"]

    if layer_params.get("reference_flash", False):
        _replace_with_reference_flash_attention(anytype, func, layer_params)
    else:
        _derive_flash_attention(anytype, func, layer_params)

    if stop_at_stage == "reduction-tiled":
        raise PipelineInterrupt()

    # Vectorize
    func = vectorize(mod, payload_func=func)

    # The accumulators of the flash loop are tensors at linalg level, so
    # vectorization turns them into a transfer_read/transfer_write pair per
    # iteration. Repeat the subset hoisting now that CSE has run, so that all
    # accumulators are carried as vector iter_args, i.e. in registers.
    reduction_loop = match(func, ops={"scf.for"})
    lh_transform.loop_hoisting(reduction_loop)

    # Turn on fast math and take the rewrite it licenses: the online rescale factor
    # arrives as `exp(-m_new) / exp(-m_old)` and becomes the single
    # `exp(m_old - m_new)` a hand-written flash-attention kernel uses, dropping a
    # transcendental and a divide per iteration. Run after vectorization -- before
    # it the two exponentials and the divide live in three separate linalg.generics,
    # so there is no single op to match.
    transform_ext.enable_fastmath_optimizations(func)
    transform.apply_cse(func)
    canonicalize(func)

    func = apply_registered_pass(func, "remove-dead-values")
    lh_transform.cleanup(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # Bufferize
    mod = bufferize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    for_all = match(mod, ops={"scf.forall"})
    func = transform.get_parent_op(anytype, for_all, op_name="func.func")

    func = convert_to_gpu_launch(mod, payload_func=func)

    # set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    num_subgroups = wg_rows // layer_params["sg_rows"]
    num_threads = num_subgroups * layer_params["subgroup_size"]
    xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])

    # outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    mod = convert_vector_to_xegpu(mod)
    lh_transform.cleanup(mod)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    gpu_mod = match(mod, ops={"gpu.module"})
    gpu_func = match(gpu_mod, ops={"gpu.func"})

    # Insert prefetches for the K and V tiles of the reduction loop. Each
    # inserts nb_prefetch prefetches ahead of the loop plus one per iteration,
    # at induction_var + nb_prefetch * step. This must run before the wg-level
    # layouts are set below, since the prefetch descriptor is cloned from the
    # load's descriptor. The layouts of the emitted prefetch_nd ops are set
    # together with the other wg-level layouts.
    nb_prefetch = layer_params.get("nb_prefetch", 1)
    if nb_prefetch > 0:
        reduction_loop = match(gpu_func, ops={"scf.for"})
        kv_load_ops = match_and_split(reduction_loop, ops={"xegpu.load_nd"}, nhandles=2)
        for load_op in kv_load_ops:
            xegpu.insert_prefetch(load_op, nb_prefetch=nb_prefetch)
        transform.apply_cse(gpu_func)
        canonicalize(gpu_func)

    # Define XeGPU layout parameters
    d_head = layer_params["d_head"]
    sg_rows = layer_params["sg_rows"]

    # Q, the attention weights and the output are all [wg_rows, d_head] tiles
    # that are split by rows over the subgroups. Only the memory ops carry
    # inst_data, the DPAS operands are left to the default DPAS blocking.
    q_sg_layout = [num_subgroups, 1]
    q_sg_data = [sg_rows, d_head]
    q_load_inst_data = layer_params["q_load_tile"]

    out_sg_layout = q_sg_layout
    out_sg_data = q_sg_data

    qk_sg_layout = q_sg_layout
    qk_sg_data = [sg_rows, reduction_tile]

    # The K and V tiles are consumed in full by every subgroup (each subgroup
    # owns its own rows of Q).
    kv_sg_layout = [1, 1]
    kv_load_sg_data = [reduction_tile, d_head]
    v_load_inst_data = layer_params["v_load_tile"]
    # Load K column-major so that the transpose feeding the DPAS is a no-op.
    k_load_order = [0, 1]

    # K^T operand of the Q@K^T DPAS: [d_head, reduction_tile]
    kt_sg_data = [d_head, reduction_tile]
    # V operand of the P@V DPAS: [reduction_tile, d_head]
    v_sg_data = [reduction_tile, d_head]

    # The K/V prefetches cover the same [reduction_tile, d_head] tile as the
    # loads, but are distributed over the subgroups.
    prefetch_sg_data = layer_params["prefetch_tile"]
    prefetch_sg_layout = [
        reduction_tile // prefetch_sg_data[0],
        d_head // prefetch_sg_data[1],
    ]
    prefetch_inst_data = list(prefetch_sg_data)

    # Set layout attributes for xegpu.store_nd ops.
    store_nd_op = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
    )

    # Set layout for xegpu.load_nd ops (3 total: Q, K, V)
    load_nd_ops = match_and_split(gpu_func, ops={"xegpu.load_nd"}, nhandles=3)

    # First load_nd: Q layout
    xegpu.set_anchor_layout(
        load_nd_ops[0],
        sg_layout=q_sg_layout,
        sg_data=q_sg_data,
        inst_data=q_load_inst_data,
    )

    # Second load_nd: K layout
    xegpu.set_anchor_layout(
        load_nd_ops[1],
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        order=k_load_order,
    )

    # Third load_nd: V layout
    xegpu.set_anchor_layout(
        load_nd_ops[2],
        sg_layout=kv_sg_layout,
        sg_data=kv_load_sg_data,
        inst_data=v_load_inst_data,
    )

    # Set layout for all K/V xegpu.prefetch_nd ops
    if nb_prefetch > 0:
        prefetch_ops = match(gpu_func, ops={"xegpu.prefetch_nd"})
        xegpu.set_anchor_layout(
            prefetch_ops,
            sg_layout=prefetch_sg_layout,
            sg_data=prefetch_sg_data,
            inst_data=prefetch_inst_data,
        )

    # Set layout for xegpu.dpas ops (2 total: Q@K^T and P@V)
    dpas_ops = match_and_split(gpu_func, ops={"xegpu.dpas"}, nhandles=2)

    # Layouts for the Q@K^T dpas:
    qk_dpas_op = dpas_ops[0]
    # Index 0: Q layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=q_sg_layout,
        sg_data=q_sg_data,
        index=0,
    )
    # Index 1: K^T layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=kt_sg_data,
        index=1,
    )
    # Index 2: QK output layout
    xegpu.set_anchor_layout(
        qk_dpas_op,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        index=2,
    )

    # Layouts for the P@V dpas:
    pv_dpas_op = dpas_ops[1]
    # Index 0: QK (attention weights) layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        index=0,
    )
    # Index 1: V layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=kv_sg_layout,
        sg_data=v_sg_data,
        index=1,
    )
    # Index 2: Output layout
    xegpu.set_anchor_layout(
        pv_dpas_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        index=2,
    )

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

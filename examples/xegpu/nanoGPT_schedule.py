"""Transform-dialect schedule for the GPU nano-GPT forward pass.

This is stage 2 -- the schedule ("how to lower it"). A "schedule" here is itself an
MLIR module written in the transform dialect: a small program of rewrite ops that
the transform interpreter runs over the payload module (built in
`nanoGPT_payload`). It does not compute anything;
it rewrites the payload from high-level linalg ops down to GPU (XeGPU) kernels.

  -> `build_combined_schedule` / `_bundle` (the orchestrator) plus the
     `_tile_one_matmul` / `_tile_one_layernorm` / `_tile_one_fused_attention_region`
     helpers.

We can't reuse the repo's per-op schedules (reduction_schedule, mlp_schedule)
directly, because each assumes the module contains only its op.
The nanoGPT module is mixed (matmul + layernorm + softmax + elementwise), so we
build one combined schedule that handles all op classes. The strategy:

  (a) Tile each op into its own parallel loop nest (`scf.forall` = the GPU
      work-group grid). Different op classes tile differently:
        - matmul   -> `_tile_one_matmul`  (work-group tile + k-loop tile; the
                       DPAS tile sizes come from `mm_params`)
        - layernorm-> `_tile_one_layernorm` (tile rows, fuse the 2 reductions +
                       2 zero-fills into the loop)
        - fused attn-> `_tile_one_fused_attention_region` (tile the chain's leaf
                       into a forall and fuse the rest in; flash loop folded after)
        - elementwise -> a single `structured_tile_using_forall` over rows
  (b) Shared tail (same for every kernel): vectorize -> bufferize (tensors ->
      memrefs) -> convert the forall grids to `gpu.launch` -> outline each into
      its own `gpu.module`/`gpu.func` kernel -> attach the XeVM target.
  (c) Annotate each kernel with XeGPU layout attributes (how data maps to
      sub-groups / DPAS tiles).

`kinds` (from the Builder) tells the schedule the class and order of every kernel,
so steps (a) and (c) can treat each one correctly.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.transform.vector import (
    apply_patterns_vector_cast_away_vector_leading_one_dim,
    apply_patterns_vector_drop_unit_dims_with_shape_cast,
)
from mlir.dialects.bufferization import LayoutMapOption

import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.xegpu.lowering_common import convert_to_gpu_launch
from lighthouse.schedule.xegpu.mlp_schedule import xegpu_wg_annotation_for_mlp_layer
from nanoGPT_payload import F32


def _tile_one_matmul(matmul_op, mm_params):
    """Tile one matmul for DPAS: a work-group `forall` tile (wg_m x wg_n) with any
    elementwise consumer fused in, then an inner reduction (k) loop. Tile sizes
    come from `mm_params` (chosen by xegpu_parameter_selector for the GPU)."""
    wg_tile = [mm_params["wg_m"], mm_params["wg_n"]]
    consumers = transform_ext.get_tileable_consumers(matmul_op)
    leaf = transform_ext.extract_handle(consumers, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    wg_matmul = match(wg_loop, ops={"linalg.matmul"})
    lh_transform.tile(wg_matmul, tile_sizes=[0, 0, mm_params["k_tile"]])


def _tile_one_layernorm(
    anytype, wg_rows, rss, mean_red, var_red, normalize, ln_untiled, n_ctx
):
    """Tile one layernorm into its own forall, using preserved handles to its 3
    generics (mean_red, var_red, normalize). Handles to other ops stay valid.

    The 2 accumulator fills are matched by their producer relationship: we match
    all fills and fuse the ones that feed this ln. To avoid touching matmul fills,
    we rely on fuse_into_containing pulling only genuine producers of the forall.
    """
    _, ln_forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        normalize,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(wg_rows,),
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=var_red, containing_op=ln_forall
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=mean_red, containing_op=ln_forall
    )
    # Fuse this ln's 2 accumulator fills into the forall. Robustly select ONLY the
    # layernorm accumulator fills (NOT matmul fills) by filtering on result type:
    # ln accumulators are rank-1 tensor<n_ctx x f32>; matmul accumulators are rank-2.
    # This avoids fragile positional counting across the whole block. There are
    # 2*ln_untiled such rank-1 fills (this ln + other untiled lns); this ln's are
    # the FIRST 2 in IR order.
    ln_func = transform.get_parent_op(
        anytype, ln_forall, op_name="func.func", deduplicate=True
    )
    reduce_t = ir.RankedTensorType.get((n_ctx,), F32())  # ln accumulator type (n_ctx,)
    fill_match = structured.MatchOp(
        anytype, ln_func, ops=["linalg.fill"], filter_result_type=reduce_t
    )
    n_ln_fills = 2 * ln_untiled
    fills = transform.split_handle((anytype,) * n_ln_fills, fill_match.results[0])
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[1], containing_op=ln_forall
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[0], containing_op=ln_forall
    )
    # Fusion leaves the full-size original fills DEAD at func scope (fusion only
    # slices a copy inside the forall). They must be removed or the next ln finds
    # too many. Use canonicalize (which does DCE of the dead originals) at func
    # scope, but never apply_cse at func scope -- CSE would merge the identical
    # live zero-fills across layernorms. CSE the duplicate generics inside the
    # forall only (scoped), so the re-match below finds exactly 3.
    transform.apply_cse(ln_forall)
    canonicalize(ln_func)
    # tile this ln's reductions+normalize (now inside the forall). Re-match the
    # 3 generics INSIDE the forall (scoped to ln_forall, so unambiguous: exactly 3).
    g2 = match_and_split(ln_forall, ops={"linalg.generic"}, nhandles=3)
    structured.TileUsingForOp(g2[2], sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[1], tile_sizes=[0, rss]
    )
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[0], tile_sizes=[0, rss]
    )
    transform.apply_cse(ln_forall)
    canonicalize(ln_forall)


def _tile_one_fused_attention_region(anytype, divide_op, fa_params):
    """Tile + fuse one attention region (QK^T -> scale -> softmax -> @V -> divide)
    into a SINGLE scf.forall, so it vectorizes/bufferizes into one kernel body that
    `_fuse_attention_in_region` then folds into the flash loop.

    Operates on a PRE-SPLIT, per-region handle (the region's deferred normalizing
    divide, which is its leaf op) so it is region-local and works at any
    multiplicity. All producers are pulled in via get_producer_of_operand
    (SSA-walk = inherently scoped to this region)."""
    prod = transform.get_producer_of_operand

    def fuse(p, c):
        return structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=p, containing_op=c
        )[1]

    wg_rows = fa_params["wg_rows"]
    # 1. Tile the region's leaf -- the `o / l` divide -- in (batch=1, M=wg_rows).
    #    It is all-parallel over (head, row, d_head), so three tile sizes.
    tiled_div, forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        divide_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_rows, 0),
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    transform.apply_cse(func)
    canonicalize(func)
    # Grab the whole producer chain UP FRONT via SSA walk (region-local; no count
    # matching). Fusing op X invalidates only X's handle, so collect all, then fuse
    # each once in consumer->producer topological order. The divide's two operands
    # are extract_slices inside the forall, so hop through each to its producer.
    o_slice = prod(anytype, tiled_div, operand_number=0)
    pv = prod(anytype, o_slice, operand_number=0)  # @V batch_matmul (unnormalized)
    l_slice = prod(anytype, tiled_div, operand_number=1)
    den = prod(anytype, l_slice, operand_number=0)  # row-sum generic
    num = prod(anytype, pv, operand_number=0)  # exp generic, feeds @V and the sum
    pv_fill = prod(anytype, pv, operand_number=2)  # 0 fill (@V acc)
    den_fill = prod(anytype, den, operand_number=1)  # 0 fill (sum acc)
    mx = prod(anytype, num, operand_number=1)  # max-reduce generic
    mx_fill = prod(anytype, mx, operand_number=1)  # -inf fill (max acc)
    scaled = prod(anytype, num, operand_number=0)  # linalg.mul (qkt*scale)
    scale_fill = prod(anytype, scaled, operand_number=1)  # scale-constant fill
    qkt = prod(anytype, scaled, operand_number=0)  # QK^T batch_matmul
    kt = prod(anytype, qkt, operand_number=1)  # K^T transpose
    qkt_fill = prod(anytype, qkt, operand_number=2)  # 0 fill (qkt acc)
    # `num` is fused after both its consumers (@V and the row sum) are inside.
    for p in (
        pv,
        den,
        num,
        mx,
        scaled,
        qkt,
        pv_fill,
        den_fill,
        mx_fill,
        scale_fill,
        qkt_fill,
        kt,
    ):
        forall = fuse(p, forall)
    transform.apply_cse(func)
    canonicalize(func)
    return func, forall


def _fuse_attention_in_region(anytype, forall, fa_params):
    """Fold one attention region's chain into the flash loop.

    `fuse_dependant_reduction_ops` does the work: it moves the elementwise term and
    one consumer reduction into an already-tiled producer reduction loop and inserts
    the online correction that rescales that reduction's running accumulator
    whenever the running max changes. Inside `forall` the chain reads:

        %s   = linalg.mul(batch_matmul(q, k^T), fill(scale))   the scaled scores
        %m   = max_j %s                                        the producer reduction
        %p   = exp(%s - %m)                                    the elementwise term
        %o   = batch_matmul(%p, v)                             consumer reduction
        %l   = sum_j %p                                        consumer reduction
        %out = %o / %l                                         the deferred divide

    Every match is scoped to `forall` so counts are exact at any multiplicity. Runs
    right after the region was tiled, i.e. still on tensors, so the shared vectorize
    tail lowers the emitted loop.

    NB: non-causal only -- there is no `causal` parameter yet.
    """
    prod = transform.get_producer_of_operand
    tile_size = fa_params["inner_loop_tile_size"]

    max_op, p_op, sum_op, _ = match_and_split(
        forall, ops={"linalg.generic"}, nhandles=4
    )
    _, pv_bmm = match_and_split(forall, ops={"linalg.batch_matmul"}, nhandles=2)
    # The fusion op wants both the elementwise term and the consumer reduction as
    # linalg.generic ops, so generalize the contraction.
    pv_op = structured.structured_generalize(anytype, pv_bmm)

    # Tile the row max along the key/value axis. This is the producer reduction loop
    # the rest of the chain gets folded into; the marker attribute is what the
    # fusion op recognizes it by.
    _, reduction_loop = structured.structured_tile_using_for(
        anytype,
        [anytype],
        max_op,
        dynamic_sizes=[],
        interchange=[],
        static_sizes=[0, 0, tile_size],
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

    # Second chain: max -> p -> @V, into that same loop. The first fusion consumed
    # the handle to `p`; the original is still the contraction's operand.
    p_op = prod(anytype, pv_op, operand_number=0)
    reduction_loop = transform_ext.fuse_dependant_reduction_ops(
        p_op, pv_op, reduction_loop
    )
    transform.apply_cse(forall)

    # Sink the score computation into the reduction loop as well, so only one
    # [wg_rows, tile_size] score tile -- rather than the full [wg_rows, n_ctx]
    # matrix -- is ever live.
    for producer_name in ["linalg.mul", "linalg.batch_matmul", "linalg.transpose"]:
        producer_op = match_and_split(forall, ops={producer_name}, nhandles=1)[0]
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
    # the fill itself. The remaining fills initialize the loop's running
    # accumulators and must stay outside.
    scale_mul_op = match_and_split(reduction_loop, ops={"linalg.mul"}, nhandles=1)[0]
    qk_matmul = match_and_split(
        reduction_loop, ops={"linalg.batch_matmul"}, nhandles=1
    )[0]
    for consumer_op, operand_number in [(scale_mul_op, 1), (qk_matmul, 2)]:
        fill_slice = prod(anytype, consumer_op, operand_number=operand_number)
        fill_op = prod(anytype, fill_slice, operand_number=0)
        _, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=fill_op,
            containing_op=reduction_loop,
        )

    transform.apply_cse(forall)
    canonicalize(forall)

    # Strip the head dim, which the region tiling cut down to 1. Everything
    # downstream -- the XeGPU layouts and the blocking/distribution passes -- is
    # built for rank-2 tiles, and the vector-level `cast_away_leading_one_dim`
    # patterns cannot finish the job: they have no pattern for multi_reduction,
    # broadcast or transpose, so the softmax row reductions and the correction's
    # broadcasts would keep a unit dim and drag shape_casts (and rank-3 XeGPU
    # layouts) along with them. `fold_unit_extent_dims` only rewrites
    # linalg.generic, hence the generalize; and it leaves two-step slice chains
    # behind, which plain canonicalization does not compose, hence the tensor
    # patterns.
    named_ops = match(
        forall,
        ops={
            "linalg.batch_matmul",
            "linalg.mul",
            "linalg.transpose",
            "linalg.fill",
            "linalg.elementwise",
        },
    )
    structured.structured_generalize(anytype, named_ops)
    with ir.InsertionPoint(transform.apply_patterns(forall).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(forall)
    with ir.InsertionPoint(transform.apply_patterns(forall).patterns):
        tensor.apply_patterns_tensor_merge_consecutive_insert_extract_slice()
        tensor.apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion()
        tensor.apply_patterns_tensor_fold_tensor_subset_ops()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(forall)


def xegpu_fa_annotation(gf, fa_params):
    """Attach XeGPU layouts to one fused-attention gpu.func."""
    num_subgroups = fa_params["wg_rows"] // fa_params["sg_rows"]
    d_head = fa_params["d_head"]
    tile_size = fa_params["inner_loop_tile_size"]
    q_sg_layout = [num_subgroups, 1]
    q_sg_data = [16, d_head]
    q_inst_data = [8, 16]
    # K and V tiles are [tile_size, d_head], shared by all subgroups.
    k_sg_layout = [num_subgroups, 1]
    k_sg_data = [tile_size, d_head]
    k_inst_data = [16, 16]
    v_sg_layout, v_sg_data, v_inst_data = k_sg_layout, k_sg_data, k_inst_data
    kt_sg_layout = [1, num_subgroups]
    kt_sg_data = [d_head, tile_size]
    kt_inst_data = [16, 16]
    kt_order = [0, 1]
    out_sg_layout, out_sg_data, out_inst_data = q_sg_layout, q_sg_data, q_inst_data
    # Q@K^T (attention weights) tile is [wg_rows, tile_size].
    qk_sg_layout = [num_subgroups, 1]
    qk_sg_data = [16, tile_size]
    qk_inst_data = [8, 16]

    store_nd_op = match_and_split(gf, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=out_inst_data,
    )
    # 3 load_nd ops: Q (hoisted out of the loop), then K and V in the loop.
    load_nd_ops = match_and_split(gf, ops={"xegpu.load_nd"}, nhandles=3)
    xegpu.set_anchor_layout(
        load_nd_ops[0], sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data
    )
    xegpu.set_anchor_layout(
        load_nd_ops[1],
        sg_layout=k_sg_layout,
        sg_data=k_sg_data,
        inst_data=k_inst_data,
    )
    xegpu.set_anchor_layout(
        load_nd_ops[2],
        sg_layout=v_sg_layout,
        sg_data=v_sg_data,
        inst_data=v_inst_data,
    )
    # 2 dpas ops: Q@K^T and P@V.
    qk_dpas, pv_dpas = match_and_split(gf, ops={"xegpu.dpas"}, nhandles=2)
    xegpu.set_anchor_layout(
        qk_dpas,
        sg_layout=q_sg_layout,
        sg_data=q_sg_data,
        inst_data=q_inst_data,
        index=0,
    )
    xegpu.set_anchor_layout(
        qk_dpas,
        sg_layout=kt_sg_layout,
        sg_data=kt_sg_data,
        inst_data=kt_inst_data,
        order=kt_order,
        index=1,
    )
    xegpu.set_anchor_layout(
        qk_dpas,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        inst_data=qk_inst_data,
        index=2,
    )
    xegpu.set_anchor_layout(
        pv_dpas,
        sg_layout=qk_sg_layout,
        sg_data=qk_sg_data,
        inst_data=qk_inst_data,
        index=0,
    )
    xegpu.set_anchor_layout(
        pv_dpas,
        sg_layout=v_sg_layout,
        sg_data=v_sg_data,
        inst_data=v_inst_data,
        index=1,
    )
    xegpu.set_anchor_layout(
        pv_dpas,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=out_inst_data,
        index=2,
    )


def build_combined_schedule(
    mm_params, ln_params, kinds, stop_at_stage="", fa_params=None
):
    """Build the transform-dialect schedule module for a payload with op classes
    `kinds`. Counts how many of each class there are, then delegates to `_bundle`
    (wrapped in transform boilerplate). `stop_at_stage` lets callers halt early
    for debugging (--dump <stage>)."""
    n_mm = kinds.count("mm")
    n_ln = kinds.count("ln")
    n_sm = kinds.count("sm")
    n_ew = kinds.count("ew")
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func0 = match(named_seq.bodyTarget, ops={"func.func"})
        mod = transform.get_parent_op(
            anytype, func0, op_name="builtin.module", deduplicate=True
        )
        try:
            _bundle(
                mod,
                mm_params,
                ln_params,
                kinds,
                n_mm,
                n_ln,
                n_sm,
                n_ew,
                stop_at_stage,
                fa_params=fa_params,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()
    return schedule


def _bundle(
    mod,
    mm_params,
    ln_params,
    kinds,
    n_mm,
    n_ln,
    n_sm,
    n_ew,
    stop_at_stage="",
    fa_params=None,
):
    """THE PASS ORCHESTRATOR -- emits the actual sequence of transform ops.

    Runs in 3 phases over the whole payload module:
      TILE   -- tile every op into a GPU work-group `forall` (per op class)
      SHARED TAIL -- vectorize, bufferize, forall->gpu.launch, outline kernels,
                     attach the XeVM target, lower vector ops to XeGPU
      ANNOTATE -- attach XeGPU sub-group/DPAS layout to each kernel
    `stop_at_stage` raises PipelineInterrupt to halt after a phase (for --dump).
    Reading the inline comments here is the best way to understand "which part of
    the code schedules the passes" -- it is this function, top to bottom."""
    anytype = transform.AnyOpType.get()
    rss = ln_params["reduction_step_size"]
    wg_rows = ln_params["wg_rows"]
    nkernels = len(kinds)
    n_fa = kinds.count("fa")

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    # ===== TILE each op-class into its own forall =====
    # Key problem: match(linalg.generic) is not scoped -- once an op is tiled into
    # a forall, its generic is still matched (it's just nested), so we can't
    # re-match "the remaining bare generics" by count. Solution: split all generic
    # handles once up front (their build order is deterministic), then tile each
    # using its preserved handle. A handle to op X stays valid across tiling of
    # OTHER ops. We tile the simple EW generics first (no fusion/cleanup, so ln
    # handles survive), then the layernorms (which fuse + cleanup).
    #
    # Generic build order: each layernorm contributes [mean, var, normalize] (3),
    # in block build order; each elementwise contributes 1. We reconstruct the
    # per-op handle slices from `kinds`.
    # Each 'fa' region contributes 4 bare generics -- row max, exp term, row sum
    # and the deferred normalizing divide (see Builder.attention_4d) -- so they ARE
    # in this pool, in that build order. The fa core's linalg.transpose /linalg.mul
    # /batch_matmul are not linalg.generic, so those stay excluded. (The head
    # reshape is a pure memref VIEW -- no generic, no kernel; see Builder.heads_view.)
    ngen_total = 3 * n_ln + n_ew + 4 * n_fa
    gen_handles = transform.split_handle(
        (anytype,) * ngen_total, match(mod, ops={"linalg.generic"})
    )
    # Walk kinds to assign generic handles to ops.
    ln_slices, ew_handles, fa_slices = [], [], []
    gi = 0
    for k in kinds:
        if k == "ln":
            ln_slices.append(
                (gen_handles[gi], gen_handles[gi + 1], gen_handles[gi + 2])
            )
            gi += 3
        elif k == "ew":
            ew_handles.append(gen_handles[gi])
            gi += 1
        elif k == "fa":
            # (row_max, probs, row_sum, divide)
            fa_slices.append(tuple(gen_handles[gi : gi + 4]))
            gi += 4
        # mm contributes no bare linalg.generic here

    # 1) Tile layernorms FIRST, using preserved (mean,var,normalize) handles.
    #    Doing this BEFORE EW/matmul tiling keeps the bare linalg.fill pool exactly
    #    predictable: 2*(untiled lns) + n_mm (matmul accumulator fills). EW tiling
    #    can introduce its own init fills, so we must finish ln fill-fusion first.
    for i, (mean_red, var_red, normalize) in enumerate(ln_slices):
        ln_untiled = n_ln - i
        _tile_one_layernorm(
            anytype,
            wg_rows,
            rss,
            mean_red,
            var_red,
            normalize,
            ln_untiled,
            ln_params["n_ctx"],
        )

    # 2) Tile EW generics into own foralls (handles preserved across ln tiling).
    for eg in ew_handles:
        structured.structured_tile_using_forall(
            anytype,
            anytype,
            eg,
            num_threads=[],
            tile_sizes=[],
            static_tile_sizes=(wg_rows,),
        )

    # 4) Matmuls (their EW producers already wrapped in foralls)
    mms = match_and_split(mod, ops={"linalg.matmul"}, nhandles=n_mm)
    for mm in mms:
        _tile_one_matmul(mm, mm_params)

    # 5) Fused-attention regions. Done last so every other kernel is already tiled
    #    when the reduction fusion runs. Each region is driven from its pre-split
    #    leaf handle -- the deferred normalizing divide -- and tiled into one forall;
    #    the flash loop is then folded out of the chain inside it.
    for divide_op in fa_slices:
        _, fa_forall = _tile_one_fused_attention_region(
            anytype, divide_op[3], fa_params
        )
        # Fold the region into the flash online-softmax loop while it is still on
        # tensors, so the shared vectorize tail lowers it like any other tiled
        # region.
        _fuse_attention_in_region(anytype, fa_forall, fa_params)

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== SHARED TAIL =====
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype, func, fold_type_extensions_into_contract=True
    )
    lh_transform.cleanup(func)
    # Accumulators of the tiled reduction loops (the flash loop's running max /
    # sum / @V accumulator, the layernorm partial reductions) are tensors at
    # linalg level, so vectorization turns them into a transfer_read/write pair
    # per iteration. Hoist those subsets so they are carried as vector iter_args,
    # i.e. in registers instead of through a scratch buffer.
    with lh_transform.foreach(match(mod, ops={"scf.for"})) as reduction_loop:
        lh_transform.loop_hoisting(reduction_loop)
        transform.yield_()
    # Each reduction fusion left the elementwise term's full-extent result as a
    # loop accumulator that nothing reads (every tile but the last is scaled by a
    # stale running max). Only now, with vectorization having replaced the in-loop
    # destination slice with transfer ops, does liveness see them as unused; left in
    # place they bufferize into real stores of a stale tensor.
    func = apply_registered_pass(func, "remove-dead-values")
    lh_transform.cleanup(func)
    # Drop any leading unit dims left over from the (1, wg_rows, 0, 0) tiling of
    # the attention regions so the QK^T/@V vector.contracts stay 2D.
    if n_fa:
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            apply_patterns_vector_cast_away_vector_leading_one_dim()
            apply_patterns_vector_drop_unit_dims_with_shape_cast()
        transform.apply_cse(func)
        canonicalize(func)
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    mod = transform_bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=True,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
    ).result
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(
        func,
        "promote-buffers-to-stack",
        options={
            "max-alloc-size-in-bytes": "8192",
            "max-rank-of-allocated-memref": "2",
        },
    )
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Shared with the per-op xegpu schedules: forall -> scf.parallel -> gpu.launch.
    func = convert_to_gpu_launch(mod, payload_func_name="payload")

    # launch threads per kernel, in IR (build) order = `kinds`.
    launches = match_and_split(mod, ops={"gpu.launch"}, nhandles=nkernels)
    mm_threads = (
        (mm_params["wg_m"] // mm_params["sg_m"])
        * (mm_params["wg_n"] // mm_params["sg_n"])
        * 16
    )
    sm_threads = (ln_params["wg_rows"] // ln_params["sg_rows"]) * ln_params[
        "subgroup_size"
    ]
    fa_threads = (
        (fa_params["wg_rows"] // fa_params["sg_rows"]) * fa_params["subgroup_size"]
        if fa_params
        else 0
    )
    for launch, kind in zip(launches, kinds):
        nt = {"mm": mm_threads, "fa": fa_threads}.get(kind, sm_threads)
        xegpu.set_gpu_launch_threads(launch, threads=[nt, 1, 1])

    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    mod = apply_registered_pass(
        mod, "xevm-attach-target", options={"O": "3", "chip": "pvc"}
    )

    # per-gpu.module convert-vector-to-xegpu. ONLY ln/sm need SLM allocas (their
    # cross-lane reductions go through shared local memory -> store_matrix). The
    # ew kernels (cast/bias/residual) are pure row-parallel: forcing their allocas
    # to SLM creates store_matrix paths that fail to lower. So SLM-ify ln/sm only;
    # leave ew (and mm) as store_nd.
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    sg_layout = [ln_params["sg_rows"], 1]
    sg_data = [ln_params["sg_rows"], rss]
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind in ("ln", "sm"):
            allocas = match(gf, ops={"memref.alloca"})
            transform_ext.update_address_space(allocas, address_space=3)
        gf = apply_registered_pass(gf, "convert-vector-to-xegpu")
        transform.apply_cse(gf)
        # Hoist loop invariants out of the kernel loops (e.g. the flash kernel
        # carries state in iter_args). apply_licm targets a loop op, so match the
        # kernel's scf.for loops and hoist each; foreach no-ops for loopless
        # (elementwise) kernels.
        with lh_transform.foreach(match(gf, ops={"scf.for"})) as k_loop:
            transform.apply_licm(k_loop)
            transform.yield_()
    transform.apply_cse(mod)
    canonicalize(mod)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # ===== PER-KERNEL ANNOTATION =====
    #   mm -> full mlp wg annotation
    #   ln -> store_nd (1) + store_matrix (the SLM reduction stores)
    #   sm -> store_nd (1) + store_matrix (4)
    #   ew -> store_nd (1) only (pure row-parallel, no SLM)
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "mm":
            xegpu_wg_annotation_for_mlp_layer(gf, **mm_params)
        elif kind == "fa":
            xegpu_fa_annotation(gf, fa_params)
        else:
            # ln/sm/ew: anchor-layout their store_nd, and (ln/sm) their SLM
            # store_matrix. Pass the whole match handle to set_anchor_layout (it
            # accepts a multi-handle) -- avoids guessing exact store counts.
            xegpu.set_anchor_layout(
                match(gf, ops={"xegpu.store_nd"}), sg_layout=sg_layout, sg_data=sg_data
            )
            if kind in ("ln", "sm"):
                xegpu.set_anchor_layout(
                    match(gf, ops={"xegpu.store_matrix"}),
                    sg_layout=sg_layout,
                    sg_data=sg_data,
                )
    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()
    return mod

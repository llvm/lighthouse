"""Transform-dialect schedule for the GPU Llama-3 forward pass.

Stage 2 -- the schedule ("how to lower it"). A "schedule" here is an MLIR module
written in the transform dialect: a program of rewrite ops that the transform
interpreter runs over the payload module (built in `llama3_payload`). It does
not compute anything; it rewrites the payload from high-level linalg ops down to
GPU (XeGPU) kernels.

  -> `build_combined_schedule` / `_bundle` (the orchestrator) plus the
     `_tile_one_matmul` / `_tile_one_rmsnorm` / `_tile_one_fused_attention_region`
     helpers.

The payload is a mixed module (matmul + RMSNorm + fused-attention + elementwise),
so this is one combined schedule that handles all op classes:

  (a) Tile each op into its own parallel loop nest (`scf.forall` = the GPU
      work-group grid). Different op classes tile differently:
        - matmul   -> `_tile_one_matmul`  (work-group tile + k-loop tile; the
                       DPAS tile sizes come from `mm_params`)
        - rmsnorm  -> `_tile_one_rmsnorm` (tile rows, fuse the reduction +
                       zero-fill into the loop)
        - fused attn-> `_tile_one_fused_attention_region` (tile @V batch_matmul into
                       a forall, fuse QK^T/scale/softmax/@V in; flash rewrite later)
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
from mlir.dialects.transform import structured, xegpu
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
from lighthouse.schedule.xegpu.mlp_schedule import xegpu_wg_annotation_for_mlp_layer
from lighthouse.schedule.xegpu.lowering_common import convert_to_gpu_launch
from llama3_payload import F32


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


def _tile_one_fused_attention_region(
    qkt_bmm: ir.Value,
    pv_bmm: ir.Value,
    softmax_op: ir.Value,
    fa_params: dict,
) -> tuple[ir.Value, ir.Value]:
    """Tile + fuse one attention region (QK^T -> scale -> softmax -> @V) into a
    single scf.forall, so it vectorizes/bufferizes into one kernel body that
    `replace_with_fused_attention` later rewrites into the flash loop.

    Operates on PRE-SPLIT, per-region handles so it is region-local and works at
    any multiplicity. All further producers are pulled in via
    get_producer_of_operand (SSA-walk = inherently scoped to this region).

    Args:
        qkt_bmm: handle of the QK^T `linalg.generic` op for this region.
        pv_bmm: handle of the attention-weights @ V `linalg.generic` op.
        softmax_op: handle of the region's (still un-decomposed) `linalg.softmax`.
        fa_params: fused-attention params (e.g. `wg_rows`).

    Returns:
        (func, forall): the enclosing `func.func` handle and the region's tiled
        `scf.forall` handle.
    """
    anytype = transform.AnyOpType.get()

    def get_producer(op, operand_number):
        """Handle of the op producing `op`'s operand (hides the anytype arg)."""
        return transform.get_producer_of_operand(
            anytype, op, operand_number=operand_number
        )

    def fuse(p, c):
        return structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=p, containing_op=c
        )[1]

    wg_rows = fa_params["wg_rows"]
    # 1. Tile the @V op (a linalg.generic for GQA) into a forall grid. The generic
    #    iterates (kv d0, rep d1, query-row i d2, head_dim l d3, key j d4); tiling
    #    kv/rep by 1 and i by wg_rows peels one query head's row block, so the inner
    #    op matches plain MHA (the fused-attention rewrite is unchanged).
    tiled_pv, forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        pv_bmm,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, 1, wg_rows, 0, 0),
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    # 2. Fuse the @V output init fill (producer of forall operand 0).
    forall = fuse(get_producer(forall, 0), forall)
    transform.apply_cse(func)
    canonicalize(func)
    # 3. Decompose this region's softmax. linalg.softmax -> 4 generics + 2 fills:
    #    max = reduce_max(scaled)         [+ -inf fill]
    #    num = exp(scaled - max)
    #    den = reduce_sum(num)            [+ 0 fill]
    #    div = num / den                  (feeds @V)
    structured.structured_decompose_interface(anytype, softmax_op)
    transform.apply_cse(func)
    canonicalize(func)
    # Grab the whole producer chain up front via SSA walk (region-local; no count
    # matching). Fusing op X invalidates only X's handle, so collect all, then fuse
    # each once in consumer->producer topological order.
    # tiled_pv operand 0 is the aw extract_slice inside the forall; hop through it
    # to the func-scope softmax `div` that it slices.
    aw_slice = get_producer(tiled_pv, 0)
    div = get_producer(aw_slice, 0)  # num / den (softmax out)
    num = get_producer(div, 0)  # exp generic
    den = get_producer(div, 1)  # sum-reduce generic
    den_fill = get_producer(den, 1)  # 0 fill (sum acc)
    mx = get_producer(num, 1)  # max-reduce generic
    mx_fill = get_producer(mx, 1)  # -inf fill (max acc)
    scaled = get_producer(num, 0)  # linalg.mul (qkt*scale)
    scale_fill = get_producer(scaled, 1)  # scale-constant fill
    qkt = get_producer(scaled, 0)  # QK^T generic
    # Q/K are block-arg views (operands 0/1), fused as loads later; operand 2 is
    # the qkt accumulator fill. No K^T transpose op: the QK^T generic contracts the
    # head-dim directly, reading K as (seq, head_dim).
    qkt_fill = get_producer(qkt, 2)  # 0 fill (qkt acc)
    for p in (
        div,
        den,
        num,
        mx,
        scaled,
        qkt,
        den_fill,
        mx_fill,
        scale_fill,
        qkt_fill,
    ):
        forall = fuse(p, forall)
    transform.apply_cse(func)
    canonicalize(func)
    return func, forall


def _fuse_attention_in_region(anytype, forall, fa_params):
    """Rewrite one attention region's tensor-level contraction pair (QK^T, @V)
    into the flash loop via the transform op. Scoped to `forall` so counts are
    exact at any multiplicity. Runs right after the region was tiled, still on
    tensors, so the shared vectorize tail lowers the emitted loop.

    `causal` (default False) makes the flash op mask future keys per query row.
    """

    def producers_by_name(target, op_names):
        return transform_ext.filter_by_name(
            transform_ext.trace_producers(target), op_names=op_names
        )

    # The two contractions (QK^T then @V) are the only contraction-like ops among
    # the region's generics (the softmax reductions are not contractions).
    linalg_ops = structured.structured_match(
        anytype, forall, ops=["linalg.generic", "linalg.batch_matmul"]
    )
    contraction_ops = transform_ext.filter_contraction_ops(linalg_ops)
    qk_matmul, pv_matmul = transform.split_handle(2 * [anytype], contraction_ops)

    # Q and K are the two tensor.extract_slice producers of QK^T (Q first); V is
    # the first extract_slice producer of @V. Passing the slices as q/k/v lets the
    # flash op recover the query-row offset for causal masking.
    qk_slices = producers_by_name(qk_matmul, "tensor.extract_slice")
    q = transform_ext.extract_handle(qk_slices, 0)
    k = transform_ext.extract_handle(qk_slices, 1)
    v = transform_ext.extract_handle(
        producers_by_name(pv_matmul, "tensor.extract_slice"), 0
    )

    # The scale is the fill value feeding the QK^T*scale mul's rhs, i.e. its
    # arith.constant. rhs (operand 1) -> linalg.fill -> constant.
    mul_op = match_and_split(forall, ops=["linalg.mul", "linalg.elementwise"], nhandles=1)[
        0
    ]
    scale = transform.get_producer_of_operand(
        anytype,
        transform.get_producer_of_operand(anytype, mul_op, operand_number=1),
        operand_number=0,
    )

    transform_ext.replace_with_fused_attention(
        q=q,
        k=k,
        v=v,
        scale=scale,
        output=pv_matmul,
        tile_size=fa_params["inner_loop_tile_size"],
        causal=fa_params.get("causal", False),
    )


def xegpu_fa_annotation(gf, fa_params):
    """Attach XeGPU layouts to one fused-attention gpu.func."""
    num_subgroups = fa_params["wg_rows"] // fa_params["sg_rows"]
    d_head = fa_params["n_head"]
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
    mm_params, ln_params, kinds, stop_at_stage="", fa_params=None, mm_params_list=None
):
    """Build the transform-dialect schedule module for a payload with op classes
    `kinds`. Counts how many of each class there are, then delegates to `_bundle`
    (wrapped in transform boilerplate). `stop_at_stage` lets callers halt early
    for debugging (--dump <stage>).

    `mm_params_list` (optional) gives per-matmul DPAS params in build order (one
    dict per 'matmul' in `kinds`); when omitted every matmul reuses `mm_params`.
    The narrow K/V projections need their own wg_n/sg_n, so the driver passes a
    list."""
    n_mm = kinds.count("matmul")
    n_rms = kinds.count("rmsnorm")
    n_sm = kinds.count("softmax")
    n_ew = kinds.count("elementwise")
    if mm_params_list is None:
        mm_params_list = [mm_params] * n_mm
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
                n_rms,
                n_sm,
                n_ew,
                stop_at_stage,
                fa_params=fa_params,
                mm_params_list=mm_params_list,
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
    n_rms,
    n_sm,
    n_ew,
    stop_at_stage="",
    fa_params=None,
    mm_params_list=None,
):
    """The pass orchestrator -- emits the actual sequence of transform ops.

    Runs in 3 phases over the whole payload module:
      tile   -- tile every op into a GPU work-group `forall` (per op class)
      shared tail -- vectorize, bufferize, forall->gpu.launch, outline kernels,
                     attach the XeVM target, lower vector ops to XeGPU
      annotate -- attach XeGPU sub-group/DPAS layout to each kernel
    `stop_at_stage` raises PipelineInterrupt to halt after a phase (for --dump)."""
    anytype = transform.AnyOpType.get()
    rss = ln_params["reduction_step_size"]
    wg_rows = ln_params["wg_rows"]
    nkernels = len(kinds)
    n_fa = kinds.count("fused_attention")
    n_rope = kinds.count("rope")
    if mm_params_list is None:
        mm_params_list = [mm_params] * n_mm

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    # ===== Tile each op-class into its own forall =====
    # match(linalg.generic) is not scoped: once an op is tiled into a forall, its
    # generic is still matched (just nested), so the remaining bare generics can't
    # be re-matched by count. Split all generic handles once up front (their build
    # order is deterministic), then tile each using its preserved handle. A handle
    # to op X stays valid across tiling of other ops. Tile the simple elementwise
    # generics first (no fusion/cleanup, so rmsnorm handles survive), then the
    # rmsnorms (which fuse + cleanup).
    #
    # Generic build order: each rmsnorm contributes [ss_sum, normed] (2), each
    # elementwise contributes 1, each RoPE contributes 1 (one multi-output generic),
    # and each fused-attention contributes [QK^T, @V] (2) -- the GQA QK^T and @V
    # contractions are linalg.generic ops (not batch_matmul) indexing the narrow
    # K/V by the outer kv dim, emitted right after the q/k/v cast ews. The slices are
    # reconstructed from `kinds`. 'fused_attention' softmax generics do not exist
    # yet (it is tiled last, softmax still un-decomposed). The attention core's
    # linalg.mul (scale) is a named op, not a generic, so excluded. (The head reshape
    # is a pure memref view -- no generic, no kernel; see Builder.heads_view.)
    ngen_total = 2 * n_rms + n_ew + n_rope + 2 * n_fa
    gen_handles = match_and_split(mod, ops={"linalg.generic"}, nhandles=ngen_total)
    # Walk kinds to assign generic handles to ops. The 'fused_attention' kinds entry
    # follows its four q/k/v/x cast ews, so gi has advanced past them and lands on
    # (QK^T, @V).
    rms_slices, ew_handles, rope_handles, fa_slices = [], [], [], []
    gi = 0
    for k in kinds:
        if k == "rmsnorm":
            rms_slices.append((gen_handles[gi], gen_handles[gi + 1]))
            gi += 2
        elif k == "elementwise":
            ew_handles.append(gen_handles[gi])
            gi += 1
        elif k == "rope":
            rope_handles.append(gen_handles[gi])
            gi += 1
        elif k == "fused_attention":
            fa_slices.append((gen_handles[gi], gen_handles[gi + 1]))
            gi += 2
        # matmul contributes no bare linalg.generic here

    # 1) Tile rmsnorms first, using preserved (ss_sum, normed) handles.
    #    Doing this before elementwise/matmul tiling keeps the bare linalg.fill pool
    #    exactly predictable: 1*(untiled rmsnorm) + n_mm (matmul accumulator fills).
    #    elementwise tiling can introduce its own init fills, so finish rmsnorm
    #    fill-fusion first.
    for i, (ss_red, normalize) in enumerate(rms_slices):
        rms_untiled = n_rms - i
        _tile_one_rmsnorm(
            mod,
            anytype,
            wg_rows,
            rss,
            ss_red,
            normalize,
            rms_untiled,
            n_mm,
            ln_params["T"],
        )

    # 2) Tile elementwise generics into own foralls (handles preserved across
    #    rmsnorm tiling).
    for eg in ew_handles:
        structured.structured_tile_using_forall(
            anytype,
            anytype,
            eg,
            num_threads=[],
            tile_sizes=[],
            static_tile_sizes=(wg_rows,),
        )

    # 3) Tile RoPE generics. Each iterates (head, T-row, coord) over a head-outer
    #    (nh,T,hs) view; tile (1, wg_rows, 0) so one grid block owns a single head's
    #    (wg_rows, half) 2D slab -> block load_nd/store_nd (see Builder.rope).
    for rg in rope_handles:
        structured.structured_tile_using_forall(
            anytype,
            anytype,
            rg,
            num_threads=[],
            tile_sizes=[],
            static_tile_sizes=(1, wg_rows, 0),
        )

    # 4) Matmuls (their EW producers already wrapped in foralls). Each matmul uses
    #    its own params (narrow K/V projections tile differently from wide matmuls).
    mms = match_and_split(mod, ops={"linalg.matmul"}, nhandles=n_mm)
    for mm, mmp in zip(mms, mm_params_list):
        _tile_one_matmul(mm, mmp)

    # 5) Fused-attention regions. Done last so the generic pre-split above ran while
    #    each fused_attention softmax was still one linalg.softmax (its decomposition generics
    #    don't exist yet, so they can't inflate ngen_total). The QK^T/@V generics
    #    use their preserved (pre-split) handles; the softmaxes still match by count.
    #    Tile+fuse each region into one forall (softmax decompose happens in-region).
    if n_fa:
        fa_softmaxes = match_and_split(mod, ops={"linalg.softmax"}, nhandles=n_fa)
        for r, (qkt_gen, pv_gen) in enumerate(fa_slices):
            _, forall = _tile_one_fused_attention_region(
                qkt_gen, pv_gen, fa_softmaxes[r], fa_params
            )
            _fuse_attention_in_region(anytype, forall, fa_params)

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== Shared tail =====
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype, func, fold_type_extensions_into_contract=True
    )
    lh_transform.cleanup(func)
    # The flash loop's accumulators (max/sum/acc) are tensors at linalg level;
    # vectorization turns them into a transfer_read/transfer_write pair per
    # iteration. Hoist them so each is carried as a vector iter_arg (in registers).
    if n_fa:
        foralls = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
        for idx, kind in enumerate(kinds):
            if kind == "fused_attention":
                lh_transform.loop_hoisting(match(foralls[idx], ops={"scf.for"}))
        lh_transform.cleanup(func)
        # Kernels tiled with a batch-of-1 head/row dim (fused attention, RoPE)
        # carry a leading unit dim; drop it so the 3D vectors collapse to the 2D
        # shapes the XeGPU layouts distribute over.
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

    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    # Shared with the per-op xegpu schedules: forall -> scf.parallel -> gpu.launch.
    func = convert_to_gpu_launch(mod, "payload")

    # launch threads per kernel, in IR (build) order = `kinds`.
    launches = match_and_split(mod, ops={"gpu.launch"}, nhandles=nkernels)

    def mm_thread_count(mmp):
        return (mmp["wg_m"] // mmp["sg_m"]) * (mmp["wg_n"] // mmp["sg_n"]) * 16

    sm_threads = (ln_params["wg_rows"] // ln_params["sg_rows"]) * ln_params[
        "subgroup_size"
    ]
    fa_threads = (
        (fa_params["wg_rows"] // fa_params["sg_rows"]) * fa_params["subgroup_size"]
        if fa_params
        else 0
    )
    mi = 0  # matmul index into mm_params_list (narrow K/V need their own threads)
    for launch, kind in zip(launches, kinds):
        if kind == "matmul":
            nt = mm_thread_count(mm_params_list[mi])
            mi += 1
        else:
            nt = {"fused_attention": fa_threads}.get(kind, sm_threads)
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

    # per-gpu.module convert-vector-to-xegpu. Only rmsnorm needs SLM allocas (its
    # cross-lane reduction goes through shared local memory -> store_matrix). The
    # elementwise kernels (cast/silu/mul/residual) are pure row-parallel: forcing
    # their allocas to SLM creates store_matrix paths that fail to lower. So SLM-ify
    # rmsnorm only; leave elementwise (and matmul) as store_nd.
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    sg_layout = [ln_params["sg_rows"], 1]
    sg_data = [ln_params["sg_rows"], rss]
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "rmsnorm":
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

    # ===== Per-kernel annotation =====
    #   matmul      -> full mlp wg annotation
    #   rmsnorm     -> store_nd (1) + store_matrix (the SLM reduction stores)
    #   elementwise -> store_nd (1) only (pure row-parallel, no SLM)
    #   rope        -> store_nd (2, the two rotated half-blocks) only, no SLM
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    mi = 0  # matmul index into mm_params_list (narrow K/V need their own layout)
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "matmul":
            xegpu_wg_annotation_for_mlp_layer(gf, **mm_params_list[mi])
            mi += 1
        elif kind == "fused_attention":
            xegpu_fa_annotation(gf, fa_params)
        else:
            # rmsnorm/elementwise/rope: anchor-layout their store_nd(s), and
            # (rmsnorm) its SLM store_matrix. Pass the whole match handle to
            # set_anchor_layout (it accepts a multi-handle) -- avoids guessing exact
            # store counts (rope has 2 store_nd, one per rotated half).
            xegpu.set_anchor_layout(
                match(gf, ops={"xegpu.store_nd"}), sg_layout=sg_layout, sg_data=sg_data
            )
            if kind == "rmsnorm":
                xegpu.set_anchor_layout(
                    match(gf, ops={"xegpu.store_matrix"}),
                    sg_layout=sg_layout,
                    sg_data=sg_data,
                )
    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()
    return mod


def _tile_one_rmsnorm(
    mod, anytype, wg_rows, rss, ss_red, normalize, rms_untiled, n_mm, T_ROWS
):
    """Tile one rmsnorm into its own forall, using preserved handles to its 2
    generics (ss_red = sum-of-squares reduction, normalize). Handles to other ops
    stay valid.

    The single accumulator fill is selected by result type: rms accumulators are
    rank-1 tensor<T x f32>; matmul accumulators are rank-2. There are rms_untiled
    such rank-1 fills (this rms + other untiled rms); this rms's is first in IR
    order.
    """
    _, rms_forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        normalize,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(wg_rows,),
    )
    _, rms_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=ss_red, containing_op=rms_forall
    )
    rms_func = transform.get_parent_op(
        anytype, rms_forall, op_name="func.func", deduplicate=True
    )
    reduce_t = ir.RankedTensorType.get((T_ROWS,), F32())  # rms accumulator type (T,)
    fill_match = structured.MatchOp(
        anytype, rms_func, ops=["linalg.fill"], filter_result_type=reduce_t
    )
    fills = transform.split_handle((anytype,) * rms_untiled, fill_match.results[0])
    if rms_untiled == 1:
        fills = [fills]  # split_handle returns a bare OpResult when nhandles==1
    _, rms_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[0], containing_op=rms_forall
    )
    # Fusion leaves the full-size original fill DEAD at func scope (fusion only
    # slices a copy inside the forall). It must be removed or the next rms finds too
    # many. Apply plain DCE at func scope -- never apply_cse at func scope, which
    # would merge the identical live zero-fills ACROSS rmsnorms. CSE the duplicate
    # generics inside the forall only (scoped), so the re-match below finds exactly 2.
    transform.apply_cse(rms_forall)
    transform.apply_dce(rms_func)
    # Re-match the 2 generics INSIDE the forall (scoped, so unambiguous: exactly 2),
    # then tile the normalize and the sum-of-squares reduction by rss.
    g2 = match_and_split(rms_forall, ops={"linalg.generic"}, nhandles=2)
    structured.TileUsingForOp(g2[1], sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[0], tile_sizes=[0, rss]
    )
    transform.apply_cse(rms_forall)
    canonicalize(rms_forall)

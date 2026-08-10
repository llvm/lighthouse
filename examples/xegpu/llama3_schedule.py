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


def _tile_one_matmul(matmul_op, anytype, mm_params):
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


def _tile_one_fused_attention_region(anytype, qkt_bmm, pv_bmm, softmax_op, fa_params):
    """Tile + fuse one attention region (QK^T -> scale -> softmax -> @V) into a
    single scf.forall, so it vectorizes/bufferizes into one kernel body that
    `replace_with_fused_attention` later rewrites into the flash loop.

    Operates on PRE-SPLIT, per-region
    handles (qkt_bmm, pv_bmm, softmax_op) so it is region-local and works at any
    multiplicity. All further producers are pulled in via get_producer_of_operand
    (SSA-walk = inherently scoped to this region)."""
    prod = transform.get_producer_of_operand

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
    forall = fuse(prod(anytype, forall, operand_number=0), forall)
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
    aw_slice = prod(anytype, tiled_pv, operand_number=0)
    div = prod(anytype, aw_slice, operand_number=0)  # num / den (softmax out)
    num = prod(anytype, div, operand_number=0)  # exp generic
    den = prod(anytype, div, operand_number=1)  # sum-reduce generic
    den_fill = prod(anytype, den, operand_number=1)  # 0 fill (sum acc)
    mx = prod(anytype, num, operand_number=1)  # max-reduce generic
    mx_fill = prod(anytype, mx, operand_number=1)  # -inf fill (max acc)
    scaled = prod(anytype, num, operand_number=0)  # linalg.mul (qkt*scale)
    scale_fill = prod(anytype, scaled, operand_number=1)  # scale-constant fill
    qkt = prod(anytype, scaled, operand_number=0)  # QK^T generic
    # Q/K are block-arg views (operands 0/1), fused as loads later; operand 2 is
    # the qkt accumulator fill. No K^T transpose op: the QK^T generic contracts the
    # head-dim directly, reading K as (seq, head_dim).
    qkt_fill = prod(anytype, qkt, operand_number=2)  # 0 fill (qkt acc)
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
    """After the shared bufferize+vectorize, rewrite one attention region's
    vector.contract pair (QK^T, @V) into the flash loop via the transform
    op. Scoped to `forall` so counts are exact at any multiplicity."""
    contract_ops = match_and_split(forall, ops={"vector.contract"}, nhandles=2)
    first_contract, second_contract = contract_ops[0], contract_ops[1]
    q_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=0
    )
    k_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=1
    )
    v_load = transform.get_producer_of_operand(
        anytype, second_contract, operand_number=1
    )
    mulf_op = match_and_split(forall, ops={"arith.mulf"}, nhandles=1)[0]
    scale = transform.get_producer_of_operand(anytype, mulf_op, operand_number=1)
    # `causal` (default False) makes the flash op mask future keys per Q row.
    transform_ext.replace_with_fused_attention(
        q_load=q_load,
        k_load=k_load,
        v_load=v_load,
        scale=scale,
        output=second_contract,
        tile_size=fa_params["inner_loop_tile_size"],
        causal=fa_params.get("causal", False),
    )


def xegpu_fa_annotation(gf, anytype, fa_params):
    """Attach XeGPU layouts to one fused-attention gpu.func."""
    num_subgroups = fa_params["wg_rows"] // fa_params["sg_rows"]
    n_head = fa_params["n_head"]
    q_sg_layout = [num_subgroups, 1]
    q_sg_data = [16, n_head]
    q_inst_data = [8, 16]
    k_sg_layout = [num_subgroups, 1]
    k_sg_data = [16, n_head]
    k_inst_data = [16, 16]
    v_sg_layout, v_sg_data, v_inst_data = k_sg_layout, k_sg_data, k_inst_data
    kt_sg_layout = [1, num_subgroups]
    kt_sg_data = [n_head, 16]
    kt_inst_data = [16, 16]
    kt_order = [0, 1]
    out_sg_layout, out_sg_data, out_inst_data = q_sg_layout, q_sg_data, q_inst_data
    l128_sg_layout = [num_subgroups, 1]
    l128_sg_data = [16, 16]
    l128_inst_data = [8, 16]
    qk_sg_layout, qk_sg_data, qk_inst_data = (
        l128_sg_layout,
        l128_sg_data,
        l128_inst_data,
    )

    store_nd_op = match_and_split(gf, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=out_inst_data,
    )
    load_nd_ops = match_and_split(gf, ops={"xegpu.load_nd"}, nhandles=9)
    xegpu.set_anchor_layout(
        load_nd_ops[0], sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data
    )
    for i in range(1, 5):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=k_sg_layout,
            sg_data=k_sg_data,
            inst_data=k_inst_data,
        )
    for i in range(5, 9):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=v_sg_layout,
            sg_data=v_sg_data,
            inst_data=v_inst_data,
        )
    dpas_ops = match_and_split(gf, ops={"xegpu.dpas"}, nhandles=8)
    for i in range(4):
        d = dpas_ops[i]
        xegpu.set_anchor_layout(
            d, sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data, index=0
        )
        xegpu.set_anchor_layout(
            d,
            sg_layout=kt_sg_layout,
            sg_data=kt_sg_data,
            inst_data=kt_inst_data,
            order=kt_order,
            index=1,
        )
        xegpu.set_anchor_layout(
            d,
            sg_layout=l128_sg_layout,
            sg_data=l128_sg_data,
            inst_data=l128_inst_data,
            index=2,
        )
    for i in range(4, 8):
        d = dpas_ops[i]
        xegpu.set_anchor_layout(
            d,
            sg_layout=qk_sg_layout,
            sg_data=qk_sg_data,
            inst_data=qk_inst_data,
            index=0,
        )
        xegpu.set_anchor_layout(
            d, sg_layout=v_sg_layout, sg_data=v_sg_data, inst_data=v_inst_data, index=1
        )
        xegpu.set_anchor_layout(
            d,
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
    dict per 'mm' in `kinds`); when omitted every matmul reuses `mm_params`. The
    narrow K/V projections need their own wg_n/sg_n, so the driver passes a list."""
    n_mm = kinds.count("mm")
    n_rms = kinds.count("rms")
    n_sm = kinds.count("sm")
    n_ew = kinds.count("ew")
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
    n_fa = kinds.count("fa")
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
    # to op X stays valid across tiling of other ops. Tile the simple ew generics
    # first (no fusion/cleanup, so rms handles survive), then the rmsnorms (which
    # fuse + cleanup).
    #
    # Generic build order: each rmsnorm contributes [ss_sum, normed] (2), each
    # elementwise contributes 1, each RoPE contributes 1 (one multi-output generic),
    # and each fused-attention contributes [QK^T, @V] (2) -- the GQA QK^T and @V
    # contractions are linalg.generic ops (not batch_matmul) indexing the narrow
    # K/V by the outer kv dim, emitted right after the q/k/v cast ews. The slices are
    # reconstructed from `kinds`. 'fa' softmax generics do not exist yet (fa is
    # tiled last, softmax still un-decomposed). The fa core's linalg.mul (scale) is a
    # named op, not a generic, so excluded. (The head reshape is a pure memref view
    # -- no generic, no kernel; see Builder.heads_view.)
    ngen_total = 2 * n_rms + n_ew + n_rope + 2 * n_fa
    gen_handles = transform.split_handle(
        (anytype,) * ngen_total, match(mod, ops={"linalg.generic"})
    )
    # Walk kinds to assign generic handles to ops. 'fa's kinds entry follows its
    # four q/k/v/x cast ews, so gi has advanced past them and lands on (QK^T, @V).
    rms_slices, ew_handles, rope_handles, fa_slices = [], [], [], []
    gi = 0
    for k in kinds:
        if k == "rms":
            rms_slices.append((gen_handles[gi], gen_handles[gi + 1]))
            gi += 2
        elif k == "ew":
            ew_handles.append(gen_handles[gi])
            gi += 1
        elif k == "rope":
            rope_handles.append(gen_handles[gi])
            gi += 1
        elif k == "fa":
            fa_slices.append((gen_handles[gi], gen_handles[gi + 1]))
            gi += 2
        # mm contributes no bare linalg.generic here

    # 1) Tile rmsnorms first, using preserved (ss_sum, normed) handles.
    #    Doing this before ew/matmul tiling keeps the bare linalg.fill pool exactly
    #    predictable: 1*(untiled rms) + n_mm (matmul accumulator fills). ew tiling
    #    can introduce its own init fills, so finish rms fill-fusion first.
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

    # 2) Tile ew generics into own foralls (handles preserved across rms tiling).
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
        _tile_one_matmul(mm, anytype, mmp)

    # 5) Fused-attention regions. Done last so the generic pre-split above ran while
    #    each fa softmax was still one linalg.softmax (its decomposition generics
    #    don't exist yet, so they can't inflate ngen_total). The QK^T/@V generics
    #    use their preserved (pre-split) handles; the softmaxes still match by count.
    #    Tile+fuse each region into one forall (softmax decompose happens in-region).
    if n_fa:
        fa_softmaxes = match_and_split(mod, ops={"linalg.softmax"}, nhandles=n_fa)
        for r, (qkt_gen, pv_gen) in enumerate(fa_slices):
            _tile_one_fused_attention_region(
                anytype, qkt_gen, pv_gen, fa_softmaxes[r], fa_params
            )

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== Shared tail =====
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype, func, fold_type_extensions_into_contract=True
    )
    lh_transform.cleanup(func)
    # Fused-attention regions carry a batch-of-1 dim from the (1,wg_rows,0,0) tiling;
    # drop leading unit dims so the QK^T/@V vector.contracts become 2D, as the flash
    # rewrite expects.
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

    # ===== Fused-attention rewrite (after bufferize+vectorize, before gpu.launch) =====
    # Re-find each attention forall by kinds index (forall IR order == kinds order,
    # the invariant the launch/gpu_mods loops below also rely on) and rewrite its
    # QK^T/@V vector.contract pair into the flash online-softmax loop. Must run
    # before forall->gpu.launch so the producer-walks for q/k/v loads stay in-region.
    if n_fa:
        all_foralls = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
        for idx, kind in enumerate(kinds):
            if kind == "fa":
                _fuse_attention_in_region(anytype, all_foralls[idx], fa_params)
        func = match(mod, ops={"func.func"})
        transform.apply_cse(func)
        canonicalize(func)
    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    # Shared with the per-op xegpu schedules: forall -> scf.parallel -> gpu.launch.
    func = convert_to_gpu_launch(mod, "payload", nlayers=nkernels)

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
        if kind == "mm":
            nt = mm_thread_count(mm_params_list[mi])
            mi += 1
        else:
            nt = {"fa": fa_threads}.get(kind, sm_threads)
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

    # per-gpu.module convert-vector-to-xegpu. Only rms needs SLM allocas (its
    # cross-lane reduction goes through shared local memory -> store_matrix). The
    # ew kernels (cast/silu/mul/residual) are pure row-parallel: forcing their
    # allocas to SLM creates store_matrix paths that fail to lower. So SLM-ify rms
    # only; leave ew (and mm) as store_nd.
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    sg_layout = [ln_params["sg_rows"], 1]
    sg_data = [ln_params["sg_rows"], rss]
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "rms":
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
    #   mm   -> full mlp wg annotation
    #   rms  -> store_nd (1) + store_matrix (the SLM reduction stores)
    #   ew   -> store_nd (1) only (pure row-parallel, no SLM)
    #   rope -> store_nd (2, the two rotated half-blocks) only, no SLM
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    mi = 0  # matmul index into mm_params_list (narrow K/V need their own layout)
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "mm":
            xegpu_wg_annotation_for_mlp_layer(gf, **mm_params_list[mi])
            mi += 1
        elif kind == "fa":
            xegpu_fa_annotation(gf, anytype, fa_params)
        else:
            # rms/ew/rope: anchor-layout their store_nd(s), and (rms) its SLM
            # store_matrix. Pass the whole match handle to set_anchor_layout (it
            # accepts a multi-handle) -- avoids guessing exact store counts (rope
            # has 2 store_nd, one per rotated half).
            xegpu.set_anchor_layout(
                match(gf, ops={"xegpu.store_nd"}), sg_layout=sg_layout, sg_data=sg_data
            )
            if kind == "rms":
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
    # many. canonicalize (DCE) at func scope, but never apply_cse at func scope --
    # CSE would merge the identical live zero-fills ACROSS rmsnorms. CSE the
    # duplicate generics inside the forall only (scoped), so the re-match below
    # finds exactly 2.
    transform.apply_cse(rms_forall)
    canonicalize(rms_func)
    # Re-match the 2 generics INSIDE the forall (scoped, so unambiguous: exactly 2),
    # then tile the normalize and the sum-of-squares reduction by rss.
    g2 = match_and_split(rms_forall, ops={"linalg.generic"}, nhandles=2)
    structured.TileUsingForOp(g2[1], sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[0], tile_sizes=[0, rss]
    )
    transform.apply_cse(rms_forall)
    canonicalize(rms_forall)

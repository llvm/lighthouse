"""Transform-dialect SCHEDULE for the GPU nano-GPT forward pass.

This is STAGE 2 -- the SCHEDULE ("HOW to lower it"). A "schedule" here is itself an
MLIR module written in the TRANSFORM dialect: a little program of rewrite ops that
the transform interpreter runs over the payload module (built in
`lighthouse.ingress.mlir_gen.gpu_nanoGPT_payload`). It does NOT compute anything;
it REWRITES the payload from high-level linalg ops down to GPU (XeGPU) kernels.

  -> `build_combined_schedule` / `_bundle` (the orchestrator) plus the
     `_tile_one_matmul` / `_tile_one_layernorm` / `_tile_one_fused_attention_region`
     helpers.

We can't reuse the repo's per-op schedules (layer_norm_schedule, mlp_schedule,
softmax_schedule) directly, because each assumes the module contains ONLY its op.
The nanoGPT module is mixed (matmul + layernorm + softmax + elementwise), so we
build ONE COMBINED schedule that handles all op classes. The strategy:

  (a) TILE each op into its own parallel loop nest (`scf.forall` = the GPU
      work-group grid). Different op classes tile differently:
        - matmul   -> `_tile_one_matmul`  (work-group tile + k-loop tile; the
                       DPAS tile sizes come from `mm_params`)
        - layernorm-> `_tile_one_layernorm` (tile rows, fuse the 2 reductions +
                       2 zero-fills into the loop)
        - fused attn-> `_tile_one_fused_attention_region` (tile @V batch_matmul into
                       a forall, fuse QK^T/scale/softmax/@V in; flash rewrite later)
        - elementwise -> a single `structured_tile_using_forall` over rows
  (b) SHARED TAIL (same for every kernel): vectorize -> bufferize (tensors ->
      memrefs) -> convert the forall grids to `gpu.launch` -> OUTLINE each into
      its own `gpu.module`/`gpu.func` kernel -> attach the XeVM target.
  (c) ANNOTATE each kernel with XeGPU layout attributes (how data maps to
      sub-groups / DPAS tiles).

`kinds` (from the Builder) tells the schedule the class and order of every kernel,
so steps (a) and (c) can treat each one correctly.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
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
from lighthouse.ingress.mlir_gen.gpu_nanoGPT_payload import F32


def _tile_one_matmul(matmul_op, anytype, mm_params):
    """Tile ONE matmul for DPAS: a work-group `forall` tile (wg_m x wg_n) with any
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
    """Tile + fuse ONE attention region (QK^T -> scale -> softmax -> @V) into a
    SINGLE scf.forall, so it vectorizes/bufferizes into one kernel body that
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
    # 1. Tile the @V batch_matmul in (batch=1, M=wg_rows) -> forall grid.
    tiled_pv, forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        pv_bmm,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_rows, 0, 0),
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
    # Grab the whole producer chain UP FRONT via SSA walk (region-local; no count
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
    qkt = prod(anytype, scaled, operand_number=0)  # QK^T batch_matmul
    kt = prod(anytype, qkt, operand_number=1)  # K^T transpose
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
        kt,
    ):
        forall = fuse(p, forall)
    transform.apply_cse(func)
    canonicalize(func)
    return func, forall


def _fuse_attention_in_region(anytype, forall, fa_params):
    """After the shared bufferize+vectorize, rewrite ONE attention region's
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
    # NB: the merged fused-attention op is non-causal only -- there is
    # no `causal` parameter yet, so the model runs as non-causal attention.
    transform_ext.replace_with_fused_attention(
        q_load=q_load,
        k_load=k_load,
        v_load=v_load,
        scale=scale,
        output=second_contract,
        tile_size=fa_params["inner_loop_tile_size"],
    )


def xegpu_fa_annotation(gf, anytype, fa_params):
    """Attach XeGPU layouts to ONE fused-attention gpu.func."""
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
    # KEY PROBLEM: match(linalg.generic) is NOT scoped -- once an op is tiled into
    # a forall, its generic is STILL matched (it's just nested), so we can't
    # re-match "the remaining bare generics" by count. SOLUTION: split ALL generic
    # handles ONCE up front (their build order is deterministic), then tile each
    # using its preserved handle. A handle to op X stays valid across tiling of
    # OTHER ops. We tile the simple EW generics first (no fusion/cleanup, so ln
    # handles survive), then the layernorms (which fuse + cleanup).
    #
    # Generic build order: each layernorm contributes [mean, var, normalize] (3),
    # in block build order; each elementwise contributes 1. We reconstruct the
    # per-op handle slices from `kinds`.
    # 'fa' softmax generics do NOT exist yet (fa is tiled last, softmax still
    # un-decomposed), so they are not in this pool. The fa core's linalg.transpose
    # /linalg.mul/batch_matmul are not linalg.generic, so also excluded. (The head
    # reshape is a pure memref VIEW -- no generic, no kernel; see Builder.heads_view.)
    ngen_total = 3 * n_ln + n_ew
    gen_handles = transform.split_handle(
        (anytype,) * ngen_total, match(mod, ops={"linalg.generic"})
    )
    # Walk kinds to assign generic handles to ops.
    ln_slices, ew_handles = [], []
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
        # mm / sm / fa contribute no bare linalg.generic here

    # 1) Tile layernorms FIRST, using preserved (mean,var,normalize) handles.
    #    Doing this BEFORE EW/matmul tiling keeps the bare linalg.fill pool exactly
    #    predictable: 2*(untiled lns) + n_mm (matmul accumulator fills). EW tiling
    #    can introduce its own init fills, so we must finish ln fill-fusion first.
    for i, (mean_red, var_red, normalize) in enumerate(ln_slices):
        ln_untiled = n_ln - i
        _tile_one_layernorm(
            mod,
            anytype,
            wg_rows,
            rss,
            mean_red,
            var_red,
            normalize,
            ln_untiled,
            n_mm,
            ln_params["T"],
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
        _tile_one_matmul(mm, anytype, mm_params)

    # 5) Fused-attention regions. Done LAST so the generic pre-split above ran while
    #    each fa softmax was still ONE linalg.softmax (its decomposition generics
    #    don't exist yet, so they can't inflate ngen_total). Pre-split the 2*n_fa
    #    batch_matmuls (build order [QK^T, @V] per region) + n_fa softmaxes by count,
    #    then tile+fuse each region into ONE forall (decompose happens in-region).
    if n_fa:
        fa_bmms = match_and_split(mod, ops={"linalg.batch_matmul"}, nhandles=2 * n_fa)
        fa_softmaxes = match_and_split(mod, ops={"linalg.softmax"}, nhandles=n_fa)
        for r in range(n_fa):
            _tile_one_fused_attention_region(
                anytype, fa_bmms[2 * r], fa_bmms[2 * r + 1], fa_softmaxes[r], fa_params
            )

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== SHARED TAIL =====
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

    # ===== FUSED-ATTENTION REWRITE (after bufferize+vectorize, before gpu.launch) =====
    # Re-find each attention forall by kinds index (forall IR order == kinds order,
    # the invariant the launch/gpu_mods loops below also rely on) and rewrite its
    # QK^T/@V vector.contract pair into the flash online-softmax loop. Must run
    # BEFORE forall->gpu.launch so the producer-walks for q/k/v loads stay in-region.
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

    wg_loops = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
    for wg_loop in wg_loops:
        loop.loop_forall_to_parallel([anytype], wg_loop)
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

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
        if kind == "fa":
            # flash kernel carries state in iter_args (no SLM); hoist invariants.
            gf = apply_registered_pass(gf, "loop-invariant-code-motion")
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
            xegpu_fa_annotation(gf, anytype, fa_params)
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


def _tile_one_layernorm(
    mod, anytype, wg_rows, rss, mean_red, var_red, normalize, ln_untiled, n_mm, T_ROWS
):
    """Tile ONE layernorm into its own forall, using PRESERVED handles to its 3
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
    # ln accumulators are rank-1 tensor<T x f32>; matmul accumulators are rank-2.
    # This avoids fragile positional counting across the whole block. There are
    # 2*ln_untiled such rank-1 fills (this ln + other untiled lns); this ln's are
    # the FIRST 2 in IR order.
    ln_func = transform.get_parent_op(
        anytype, ln_forall, op_name="func.func", deduplicate=True
    )
    reduce_t = ir.RankedTensorType.get((T_ROWS,), F32())  # ln accumulator type (T,)
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
    # too many. Use canonicalize (which does DCE of the dead originals) at FUNC
    # scope, but NEVER apply_cse at func scope -- CSE would merge the identical
    # live zero-fills ACROSS layernorms. CSE the duplicate GENERICS inside the
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

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
        - fused attn-> `_tile_one_fused_attention_region` (tile the attention
                       output, fuse its producers, then fuse the reductions)
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
from lighthouse.schedule.xegpu.attention_fusion import annotate_fastmath_flags
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


def _tile_one_fused_attention_region(anytype, output_op, fa_params):
    """Tile one attention output and fuse its explicit producer chain."""
    prod = transform.get_producer_of_operand

    def producer(op, operand):
        return prod(anytype, op, operand_number=operand)

    def fuse(op, loop):
        return structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=op, containing_op=loop
        )[1]

    _, forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        output_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, 1, fa_params["wg_rows"], 0),
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    transform.apply_cse(func)
    canonicalize(func)

    # The output's numerator and denominator enter the forall through slices.
    tiled_output = match_and_split(forall, ops={"linalg.generic"}, nhandles=1)[0]
    numerator = producer(producer(tiled_output, 0), 0)
    row_sum = producer(producer(tiled_output, 1), 0)
    probs = producer(numerator, 0)
    numerator_fill = producer(numerator, 2)
    sum_fill = producer(row_sum, 1)
    row_max = producer(probs, 1)
    max_fill = producer(row_max, 1)
    scores = producer(probs, 0)
    mask = scores if fa_params["causal"] else None
    scale = producer(mask, 0) if mask is not None else scores
    scale_fill = producer(scale, 1)
    qkt = producer(scale, 0)
    qkt_fill = producer(qkt, 2)

    # Fuse both consumers of probs before probs itself.
    for op in (
        numerator,
        row_sum,
        probs,
        row_max,
        mask,
        scale,
        qkt,
        numerator_fill,
        sum_fill,
        max_fill,
        scale_fill,
        qkt_fill,
    ):
        if op is not None:
            forall = fuse(op, forall)
    transform.apply_cse(func)
    canonicalize(func)
    return func, forall


def _fuse_attention_in_region(anytype, forall, fa_params):
    """Fuse row sum and P@V into the tiled max loop of one GQA region."""
    prod = transform.get_producer_of_operand
    reductions = transform_ext.filter_reduction_ops(
        match(forall, ops={"linalg.generic"})
    )
    qkt, row_max, row_sum, pv = transform.split_handle([anytype] * 4, reductions)

    probs = prod(anytype, pv, operand_number=0)
    score = prod(anytype, row_max, operand_number=0)
    mask = score if fa_params["causal"] else None
    scale = prod(anytype, mask, operand_number=0) if mask is not None else score

    _, loop = structured.structured_tile_using_for(
        anytype,
        [anytype],
        row_max,
        dynamic_sizes=[],
        interchange=[],
        static_sizes=[0, 0, 0, fa_params["inner_loop_tile_size"]],
        scalable_sizes=[False] * 4,
    )
    transform.annotate(loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

    # P@V first leaves the final loop state in max, sum, numerator order;
    # that order also lets XeGPU legalize the causal index vectors.
    loop = transform_ext.fuse_dependent_reduction_ops(probs, pv, loop)
    probs = prod(anytype, row_sum, operand_number=0)
    loop = transform_ext.fuse_dependent_reduction_ops(probs, row_sum, loop)
    transform.apply_cse(forall)

    # Bring the masked score and its producers into the key-tile loop.
    if mask is not None:
        _, loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, mask, loop
        )
    tiled_scale, loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, scale, loop
    )
    tiled_qkt, loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, qkt, loop
    )
    for consumer, operand in ((tiled_scale, 1), (tiled_qkt, 2)):
        fill_slice = prod(anytype, consumer, operand_number=operand)
        fill = prod(anytype, fill_slice, operand_number=0)
        _, loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, fill, loop
        )

    transform.apply_cse(forall)
    canonicalize(forall)
    named = match(forall, ops={"linalg.fill", "linalg.elementwise"})
    structured.structured_generalize(anytype, named)
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
    annotate_fastmath_flags(forall)


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
    # Each attention region contributes QK, optional mask, max, exp, sum,
    # P@V and the final divide/cast. Scale remains a named elementwise op.
    fa_generics = 6 + int(fa_params["causal"]) if n_fa else 0
    ngen_total = 2 * n_rms + n_ew + n_rope + fa_generics * n_fa
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
            fa_slices.append(gen_handles[gi + fa_generics - 1])
            gi += fa_generics
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

    # Tile each attention output, then fuse its max -> exp -> (sum, P@V) chain.
    for output_op in fa_slices:
        _, forall = _tile_one_fused_attention_region(anytype, output_op, fa_params)
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
        func = apply_registered_pass(func, "remove-dead-values")
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
    func = convert_to_gpu_launch(mod, payload_func_name="payload")

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

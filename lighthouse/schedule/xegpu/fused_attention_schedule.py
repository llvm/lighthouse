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

    In step 2., the inner attention block is tiled and fused over the
    reduction dimension (n_ctx) of the final P@V operation, controlled by the
    `reduction_tile` parameter. The Q@K^T and softmax operations are fused into
    the P@V loop, implementing online softmax. This happens at tensor level, so
    the tiling and fusion decisions stay at the level the rest of the schedule
    works on and the emitted loop is lowered by the regular vectorization step.

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

    # Match linalg.softmax operation if any and decompose it into generic ops
    softmax_ops = structured.structured_match(anytype, func, ops=["linalg.softmax"])
    structured.structured_decompose_interface(anytype, softmax_ops)
    # Convert linalg.mul and linalg.batch_matmul to linalg.generic
    structured.structured_generalize(
        anytype,
        structured.structured_match(
            anytype, func, ops=["linalg.mul", "linalg.batch_matmul"]
        ),
    )

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

    # Apply reduction tiling and fusion, still at tensor level. The Q, K, V
    # tensors and the scale constant are found by walking the SSA chain of the
    # two batch matmuls inside the WG forall:
    #
    #   Q@K^T:  linalg.batch_matmul(q_slice, linalg.transpose(k_slice))
    #   scale:  linalg.mul(qkt, linalg.fill(scale_constant))
    #   P@V:    linalg.batch_matmul(softmax_out, v_slice)

    linalg_ops = structured.structured_match(
        anytype, func, ops=["linalg.generic", "linalg.batch_matmul"]
    )
    contraction_ops = transform_ext.filter_contraction_ops(linalg_ops)

    # Match max reduction op. Assumes there's only one arith.max* op.
    arith_max_op = match_and_split(
        func, ops=["arith.maximumf", "arith.maxnumf"], nhandles=1
    )[0]
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
    max_producer = transform_ext.extract_handle(max_producer_generics, 0)
    max_scale_mul_op = match_and_split(max_producer, ops={"arith.mulf"}, nhandles=1)[0]
    scale_producers = transform_ext.trace_producers(max_scale_mul_op)
    scale_const_op = transform_ext.extract_handle(
        transform_ext.filter_by_name(scale_producers, op_names="arith.constant"), 0
    )

    matmul_ops = transform.split_handle(2 * [anytype], contraction_ops)
    qk_matmul, pv_matmul = matmul_ops[0], matmul_ops[1]

    # Find the tensor.extract_slice producers for the Q@K^T matmul.
    qk_extract_slice_producers = get_producers_by_name(
        qk_matmul, op_names="tensor.extract_slice"
    )
    q = transform_ext.extract_handle(qk_extract_slice_producers, 0)
    k = transform_ext.extract_handle(qk_extract_slice_producers, 1)

    # Find handle to v as the first tensor.extract_slice producer of PV matmul
    pv_extract_slice_producers = get_producers_by_name(
        pv_matmul, op_names="tensor.extract_slice"
    )
    v = transform_ext.extract_handle(pv_extract_slice_producers, 0)

    # Replace the P@V batch matmul with a loop over the K/V sequence length that
    # implements online softmax, fusing Q@K^T and the softmax into it.
    reduction_tile = layer_params[
        "reduction_tile"
    ]  # Tile size for reduction dimension (K/V sequence length)
    transform_ext.replace_with_fused_attention(
        q=q,
        k=k,
        v=v,
        scale=scale_const_op,
        output=pv_matmul,
        tile_size=reduction_tile,
    )
    transform.apply_cse(func)
    lh_transform.cleanup(func)

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

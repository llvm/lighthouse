"""Generate MLIR transform schedule for XeGPU fused attention operation."""

from typing import Optional

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform.vector import (
    apply_patterns_vector_cast_away_vector_leading_one_dim,
    apply_patterns_vector_drop_unit_dims_with_shape_cast,
)

from lighthouse.pipeline.helper import (
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
    apply_registered_pass,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform.transform_ext import (
    generate_fused_attention,
    update_address_space,
)


def fused_attention_schedule(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> ir.Module:
    """
    Generate transform schedule for fused attention operation.

    The schedule performs the following transformations:
    1. Tile the fused attention operation
    2. Vectorize operations
    3. Bufferize tensors
    4. Convert to GPU dialect
    5. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - batch_size: Batch size (Z)
            - num_heads: Number of attention heads (H)
            - n_ctx: Context length
            - n_head: Head dimension
            - wg_rows: Number of Q*K^T*V rows computed by each work group
            - sg_rows: Number of Q*K^T*V rows computed by each subgroup
            - subgroup_size: Size of subgroup

    Returns:
        MLIR module containing the transform schedule
    """
    assert parameters is not None, "Schedule parameters must be provided"

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
                parameters=parameters,
                stop_at_stage=stop_at_stage or "",
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_fused_attention_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering fused attention payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()
    # Match all matmul operations - there should be 2:
    # 1. Q @ K^T
    # 2. attention_weights @ V
    matmul_ops = match_and_split(mod, ops={"linalg.batch_matmul"}, nhandles=2)

    # Get the last matmul (attention_weights @ V)
    last_matmul = matmul_ops[1]
    func = transform.get_parent_op(
        anytype,
        last_matmul,
        op_name="func.func",
        deduplicate=True,
    )

    # Tile the last matmul in both batch and M dimensions.
    wg_rows = parameters["wg_rows"]

    tiled_matmul, forall_loop = structured.structured_tile_using_forall(
        anytype,
        anytype,
        last_matmul,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_rows, 0, 0),
    )
    # Fuse the zero initialization of the output of the last matmul (tensor.empty) into the forall loop.
    tiled_matmul_init = transform.get_producer_of_operand(
        anytype, forall_loop, operand_number=0
    )
    _, forall_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=tiled_matmul_init,
        containing_op=forall_loop,
    )
    transform.apply_cse(func)
    canonicalize(func)

    # Decompose softmax into generic ops
    softmax_ops = match_and_split(func, ops={"linalg.softmax"}, nhandles=1)
    softmax_op = softmax_ops[0]
    structured.structured_decompose_interface(anytype, softmax_op)
    transform.apply_cse(func)
    canonicalize(func)

    # Fuse all linalg.generic ops from softmax decomposition (4 ops: max, sub+exp, sum, div)
    # Match and fuse in reverse order (from consumer to producer)
    generic_ops = match_and_split(func, ops={"linalg.generic"}, nhandles=4)
    for generic_op in reversed(generic_ops):
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=generic_op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    # Max and add reductions use linalg.fill to intialize the reduction output. Fuse these fill ops as well.
    fill_ops = match_and_split(func, ops={"linalg.fill"}, nhandles=5)
    # Max fill is the third fill op and add fill is the fourth fill op (based on the pattern of decomposition)
    max_fill_op = fill_ops[2]
    add_fill_op = fill_ops[3]
    for fill_op in [max_fill_op, add_fill_op]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=fill_op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    linalg_mul_op = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]
    first_matmul = transform.get_producer_of_operand(
        anytype, linalg_mul_op, operand_number=0
    )
    scale_fill_op = transform.get_producer_of_operand(
        anytype, linalg_mul_op, operand_number=1
    )
    transpose_op = transform.get_producer_of_operand(
        anytype, first_matmul, operand_number=1
    )
    matmul_fill_op = transform.get_producer_of_operand(
        anytype, first_matmul, operand_number=2
    )
    for op in [
        linalg_mul_op,
        scale_fill_op,
        first_matmul,
        matmul_fill_op,
        transpose_op,
    ]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=op,
            containing_op=forall_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "outer-tiled":
        raise PipelineInterrupt()

    # vectorize
    func = structured.VectorizeChildrenAndApplyPatternsOp(
        func,
        fold_type_extensions_into_contract=True,
    ).result
    transform.apply_cse(func)
    canonicalize(func)
    # Try to remove any unit dimensions that may have been introduced due to tiling (e.g. batch dim of 1)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        apply_patterns_vector_cast_away_vector_leading_one_dim()
        apply_patterns_vector_drop_unit_dims_with_shape_cast()

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize
    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    identity_layout = LayoutMapOption.IdentityLayoutMap
    mod = transform_bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=True,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=identity_layout,
    ).result
    transform.apply_cse(mod)
    canonicalize(mod)
    # fold memref.subviews into vector.transfer_read/write ops
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    # promote memref.alloc to memref.alloca in payload function
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(
        func,
        "promote-buffers-to-stack",
        options={
            "max-alloc-size-in-bytes": "8192",
            "max-rank-of-allocated-memref": "2",
        },
    )

    # Extract q, k, v memrefs from the bufferized IR
    # Match vector.contract ops to find the q, k, v loads
    for_all = match(mod, ops={"scf.forall"})
    func = transform.get_parent_op(anytype, for_all, op_name="func.func")
    contract_ops = match_and_split(func, ops={"vector.contract"}, nhandles=2)

    # First vector.contract is Q @ K^T
    # Its first operand is the q load (vector.transfer_read)
    # Its second operand is the k load (vector.transfer_read)
    first_contract = contract_ops[0]
    q_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=0
    )
    k_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=1
    )

    # # Second vector.contract is attention_weights @ V
    # # Its second operand is the v load (vector.transfer_read)
    second_contract = contract_ops[1]
    v_load = transform.get_producer_of_operand(
        anytype, second_contract, operand_number=1
    )

    # Match arith.mulf to get the scale parameter
    # The scale is the second operand of arith.mulf (the constant)
    mulf_op = match_and_split(func, ops={"arith.mulf"}, nhandles=1)[0]
    scale = transform.get_producer_of_operand(anytype, mulf_op, operand_number=1)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Generate fused attention computation with inner tiling
    # This replaces the second vector.contract (attention_weights @ V) with a tiled
    # loop that implements online softmax for efficient memory usage
    tile_size = parameters.get(
        "inner_loop_tile_size", 64
    )  # Tile size for reduction dimension (K/V sequence length)
    generate_fused_attention(
        q_load=q_load,
        k_load=k_load,
        v_load=v_load,
        scale=scale,
        output=second_contract,
        tile_size=tile_size,
    )
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    # convert forall to parallel
    wg_loops = match_and_split(mod, ops={"scf.forall"})
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # convert scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    wg_rows = parameters["wg_rows"]
    sg_rows = parameters["sg_rows"]
    subgroup_size = parameters["subgroup_size"]
    num_subgroups = wg_rows // sg_rows
    num_threads = num_subgroups * subgroup_size
    xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])

    # outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    # set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # for each gpu function in the gpu module, change memref.alloca address
    # space to 3 (SLM) and convert vector to xegpu.
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"})
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        allocas = match_and_split(gpu_func, ops={"memref.alloca"}, nhandles=3)
        for alloca in allocas:
            # print("Updating address space for alloca:")
            update_address_space(alloca, address_space=3)
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)
        gpu_func = apply_registered_pass(gpu_func, "loop-invariant-code-motion")

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Define XeGPU layout parameters
    q_sg_layout = [8, 1]
    q_sg_data = [16, 64]
    q_inst_data = [8, 16]

    k_sg_layout = [8, 1]
    k_sg_data = [16, 64]
    k_inst_data = [16, 16]

    v_sg_layout = k_sg_layout
    v_sg_data = k_sg_data
    v_inst_data = k_inst_data

    kt_sg_layout = [1, 8]
    kt_sg_data = [64, 16]
    kt_inst_data = [16, 16]
    kt_order = [0, 1]

    out_sg_layout = q_sg_layout
    out_sg_data = q_sg_data
    out_inst_data = q_inst_data

    layout_128x16_sg_layout = [8, 1]
    layout_128x16_sg_data = [16, 16]
    layout_128x16_inst_data = [8, 16]

    qk_sg_layout = layout_128x16_sg_layout
    qk_sg_data = layout_128x16_sg_data
    qk_inst_data = layout_128x16_inst_data

    # Set layout attributes for xegpu.store_nd ops.
    store_nd_op = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=out_inst_data,
    )

    # Set layout for xegpu.load_nd ops (9 total: 1 Q, 4 K, 4 V)
    load_nd_ops = match_and_split(gpu_func, ops={"xegpu.load_nd"}, nhandles=9)

    # First load_nd: Q layout
    xegpu.set_anchor_layout(
        load_nd_ops[0], sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data
    )

    # Next 4 load_nd ops: K layout
    for i in range(1, 5):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=k_sg_layout,
            sg_data=k_sg_data,
            inst_data=k_inst_data,
        )

    # Last 4 load_nd ops: V layout
    for i in range(5, 9):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=v_sg_layout,
            sg_data=v_sg_data,
            inst_data=v_inst_data,
        )

    # Set layout for xegpu.dpas ops (8 total: 4 for Q@K, 4 for P@V)
    dpas_ops = match_and_split(gpu_func, ops={"xegpu.dpas"}, nhandles=8)

    # Layouts for first 4 dpas ops (Q@K^T):
    for i in range(4):
        qk_dpas_op = dpas_ops[i]
        # Index 0: Q layout
        xegpu.set_anchor_layout(
            qk_dpas_op,
            sg_layout=q_sg_layout,
            sg_data=q_sg_data,
            inst_data=q_inst_data,
            index=0,
        )
        # Index 1: K^T layout
        xegpu.set_anchor_layout(
            qk_dpas_op,
            sg_layout=kt_sg_layout,
            sg_data=kt_sg_data,
            inst_data=kt_inst_data,
            order=kt_order,
            index=1,
        )
        # Index 2: QK output layout (128x16)
        xegpu.set_anchor_layout(
            qk_dpas_op,
            sg_layout=layout_128x16_sg_layout,
            sg_data=layout_128x16_sg_data,
            inst_data=layout_128x16_inst_data,
            index=2,
        )

    # Layouts for second 4 dpas ops (P@V):
    for i in range(4, 8):
        pv_dpas_op = dpas_ops[i]
        # Index 0: QK (attention weights) layout
        xegpu.set_anchor_layout(
            pv_dpas_op,
            sg_layout=qk_sg_layout,
            sg_data=qk_sg_data,
            inst_data=qk_inst_data,
            index=0,
        )
        # Index 1: V layout
        xegpu.set_anchor_layout(
            pv_dpas_op,
            sg_layout=v_sg_layout,
            sg_data=v_sg_data,
            inst_data=v_inst_data,
            index=1,
        )
        # Index 2: Output layout
        xegpu.set_anchor_layout(
            pv_dpas_op,
            sg_layout=out_sg_layout,
            sg_data=out_sg_data,
            inst_data=out_inst_data,
            index=2,
        )

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

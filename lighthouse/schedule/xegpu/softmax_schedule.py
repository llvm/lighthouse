"""Generate MLIR transform schedule for XeGPU softmax operation."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption

from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform import transform_ext


def softmax_schedule(
    stop_at_stage: str | None = None,
    parameters: dict | None = None,
) -> ir.Module:
    """
    Generate transform schedule for softmax operation.

    The schedule performs the following transformations:
    1. Tile the linalg.softmax operation using forall
    2. Vectorize operations
    3. Bufferize tensors
    4. Convert to GPU dialect
    5. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - wg_rows: Number of rows per workgroup
            - sg_rows: Number of rows per subgroup
            - subgroup_size: Size of subgroup
            - sizes: Tuple with the sizes of the input tensors (e.g. (M, N))
            - reduction_step_size: Optional step size for tiling reduction loops

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
            bundle_xegpu_softmax_schedule(
                payload_mod,
                parameters=parameters,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_softmax_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering softmax payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # Match linalg.softmax operation
    # We have only 1 operation: linalg.softmax
    softmax_op = structured.structured_match(
        transform.AnyOpType.get(), mod, ops=["linalg.softmax"]
    )

    # Tile the softmax operation using tile_using_forall
    tiled_op, for_op = structured.structured_tile_using_forall(
        anytype,
        anytype,
        softmax_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(parameters["wg_rows"],),
    )

    func = transform.get_parent_op(
        anytype,
        for_op,
        op_name="func.func",
        deduplicate=True,
    )
    # Decompose softmax into linalg.generic operations
    softmax_ops = structured.structured_match(
        transform.AnyOpType.get(), func, ops=["linalg.softmax"]
    )
    structured.structured_decompose_interface(anytype, softmax_ops)

    linalg_ops = match_and_split(
        func, ops={"linalg.generic", "linalg.fill"}, nhandles=6
    )
    max_reduction = linalg_ops[1]
    max_center_and_exp_op = linalg_ops[2]
    sum_reduction = linalg_ops[4]
    div_op = linalg_ops[5]

    reduction_step_size = parameters["reduction_step_size"]

    # Tile the division op first.
    _, div_loop = structured.TileUsingForOp(
        div_op, sizes=[0, reduction_step_size]
    ).results

    # Fuse max_center_and_exp_op into the div loop
    _, fused_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=max_center_and_exp_op,
        containing_op=div_loop,
    )

    # Tile the sum reduction.
    _, _, _, sum_loop = structured.structured_tile_reduction_using_for(
        [anytype],
        anytype,
        anytype,
        anytype,
        target=sum_reduction,
        tile_sizes=[0, reduction_step_size],
    )

    func = transform.get_parent_op(
        anytype,
        fused_loop,
        op_name="func.func",
        deduplicate=True,
    )

    # Re-match and split linalg generic ops, there are 5 at this point
    linalg_ops = match_and_split(func, ops={"linalg.generic"}, nhandles=5)
    max_center_and_exp_op = linalg_ops[1]

    # Fuse max_center_and_exp_op into the sum reduction loop
    _, fused_sum_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=max_center_and_exp_op,
        containing_op=sum_loop,
    )

    # Tile the max reduction.
    max_reduction = linalg_ops[0]
    structured.structured_tile_reduction_using_for(
        [anytype],
        anytype,
        anytype,
        anytype,
        target=max_reduction,
        tile_sizes=[0, reduction_step_size],
    )

    # Cleanup after tiling and fusion
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize
    func = structured.VectorizeChildrenAndApplyPatternsOp(
        func,
        fold_type_extensions_into_contract=True,
    ).result
    transform.apply_cse(func)
    canonicalize(func)

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

    if stop_at_stage == "bufferized":
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
    num_subgroups = parameters["wg_rows"] // parameters["sg_rows"]
    num_threads = num_subgroups * parameters["subgroup_size"]
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
        allocas = match(gpu_func, ops={"memref.alloca"})
        transform_ext.update_address_space(allocas, address_space=3)
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)

    # Cleanup.
    transform.apply_cse(mod)
    canonicalize(mod)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Set layout attributes for xegpu.store_nd and xegpu.store_matrix ops.
    store_nd_ops = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)
    store_matrix_ops = match_and_split(gpu_func, ops={"xegpu.store_matrix"}, nhandles=4)
    sg_layout = [parameters["sg_rows"], 1]
    sg_data = [parameters["sg_rows"], parameters["reduction_step_size"]]
    for store_op in store_nd_ops:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
    for store_op in store_matrix_ops:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

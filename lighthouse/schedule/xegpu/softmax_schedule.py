"""Generate MLIR transform schedule for XeGPU softmax operation."""

from typing import Optional

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
from lighthouse.schedule.xegpu.helper import bundle_xegpu_to_binary


def get_softmax_schedule_module(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
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

    Returns:
        MLIR module containing the transform schedule
    """
    assert parameters is not None, "Schedule parameters must be provided"

    mod = ir.Module.create()
    mod.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()

    with ir.InsertionPoint(mod.body):
        # Create a transform sequence with proper signature
        named_sequence = transform.named_sequence(
            "__transform_main",
            [transform.AnyOpType.get()],  # input: module
            [],  # no outputs
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )

        with ir.InsertionPoint(named_sequence.body):
            # match the payload module
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            payload_mod = transform.get_parent_op(
                anytype,
                func,
                op_name="builtin.module",
                deduplicate=True,
            )

            xegpu_softmax_transform_schedule(
                payload_mod,
                parameters=parameters,
                stop_at_stage=stop_at_stage or "",
            )

    return mod


def xegpu_softmax_transform_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
):
    """Transform schedule for softmax payload."""
    try:
        mod = bundle_xegpu_softmax_schedule(
            mod,
            parameters=parameters,
            stop_at_stage=stop_at_stage,
        )

        mod = bundle_xegpu_to_binary(
            mod,
            stop_at_stage=stop_at_stage,
        )
    except PipelineInterrupt:
        pass
    finally:
        transform.yield_()


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

    # set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # convert vector to xegpu
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"})
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Set layout attributes for xegpu.store_nd operations.
    # FIXME: currently ecah subgroup is handling the entire row.
    store_ops = match_and_split(gpu_func, ops={"xegpu.store_nd"}, nhandles=1)
    sg_layout = [parameters["sg_rows"], 1]
    sg_data = [parameters["sg_rows"], parameters["sizes"][1]]
    xegpu.set_anchor_layout(store_ops[0], sg_layout=sg_layout, sg_data=sg_data)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

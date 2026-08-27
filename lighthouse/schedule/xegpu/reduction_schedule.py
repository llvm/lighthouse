"""Generate MLIR transform schedule for XeGPU softmax operation."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, xegpu, tensor
import lighthouse.transform as lh_transform
from .lowering_common import (
    get_payload_func,
    vectorize,
    bufferize,
    convert_to_gpu_launch,
    convert_vector_to_xegpu,
)
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.schedule.parameters import ScheduleParameters
from lighthouse.dialects.transform import transform_ext


def reduction_schedule(
    stop_at_stage: str | None = None,
    params: ScheduleParameters | None = None,
    payload_func_name: str = "payload",
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
        params: ScheduleParameters object containing one reduction layer
            parameter dictionary with keys:
            - wg_rows: Number of rows per workgroup
            - sg_rows: Number of rows per subgroup
            - subgroup_size: Size of subgroup
            - sizes: Tuple with the sizes of the input tensors (e.g. (M, N))
            - reduction_step_size: Optional step size for tiling reduction loops

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
            bundle_xegpu_reduction_schedule(
                payload_mod,
                payload_func_name=payload_func_name,
                params=params,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_reduction_schedule(
    mod: ir.Value[transform.AnyOpType],
    payload_func_name: str,
    params: ScheduleParameters,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering softmax payload to xegpu wg level."""

    layer_params = params[0]

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    reduction_step_size = layer_params["reduction_step_size"]

    anytype = transform.AnyOpType.get()

    # Match linalg.softmax operation if any and decompose it into generic ops
    softmax_ops = structured.structured_match(anytype, mod, ops=["linalg.softmax"])
    structured.structured_decompose_interface(anytype, softmax_ops)

    # Match payload function
    # TODO match with given function name instead?
    generic_ops = structured.structured_match(anytype, mod, ops=["linalg.generic"])
    func = transform.get_parent_op(
        anytype,
        generic_ops,
        op_name="func.func",
        deduplicate=True,
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

    # WG row tiling
    generic_ops = structured.structured_match(anytype, mod, ops=["linalg.generic"])
    leaf_generic = transform_ext.extract_handle(generic_ops, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf_generic,
        tile_sizes=(layer_params["wg_rows"],),
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)
    wg_loop = match_and_split(func, ops={"scf.forall"}, nhandles=1)[0]

    # Reduction dimension tiling.
    # 1. Tile the leaf elemwise linalg.generic op and fuse its elemwise
    #    linalg.generic producers into the resulting loop.
    # 2. Tile each reduction linalg.generic op (from last to first) and fuse its
    #    elemwise producers into the resulting loop.

    def fuse_elemwise_producers_to_loop(target, parent_loop):
        """Fuses all elementwise producer ops of `target` into `parent_loop`."""
        producers = transform_ext.trace_producers(target)
        elemwise_producers = transform_ext.filter_elementwise(producers)
        elemwise_producers = transform_ext.filter_by_name(
            elemwise_producers,
            "linalg.generic",
        )
        _, fused_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=elemwise_producers,
            containing_op=parent_loop,
        )
        return fused_loop

    generic_ops = match(wg_loop, ops={"linalg.generic"})
    elemwise_ops = transform_ext.filter_elementwise(generic_ops)
    leaf_elemwise = transform_ext.extract_handle(elemwise_ops, -1)
    reduction_ops = transform_ext.filter_reduction_ops(generic_ops)

    reduction_tile_size = [0, reduction_step_size]

    # Tile trailing elemwise op first.
    tiled_elemwise, tile_loop = structured.TileUsingForOp(
        leaf_elemwise, sizes=reduction_tile_size
    ).results
    # Fuse all elemwise producers into the tiled leaf loop.
    fuse_elemwise_producers_to_loop(tiled_elemwise, tile_loop)

    def tile_and_fuse_reduction(reduction_op, tile_sizes):
        # Tile the reduction op.
        _, tiled_op, _, tile_loop = structured.structured_tile_reduction_using_for(
            [anytype],
            anytype,
            anytype,
            anytype,
            target=reduction_op,
            tile_sizes=tile_sizes,
        )
        # Fuse all elemwise producers into the tiled leaf loop.
        fuse_elemwise_producers_to_loop(tiled_op, tile_loop)

    # Tile and fuse the reduction loops in reverse order. After each fusion
    # step, DCE removes the dead untiled elementwise epilogue so it cannot
    # create a cross-loop use that breaks the next tile-fuse iteration. Note
    # that DCE does not invalidate the reduction loop handles as the tracking
    # listener only invalidates modified handles and the reduction loops are
    # alive and thus not removed.
    reduction_ops = transform_ext.reverse_handles(reduction_ops)
    with lh_transform.foreach(reduction_ops) as reduction_op:
        tile_and_fuse_reduction(reduction_op, reduction_tile_size)
        transform.apply_dce(wg_loop)
        transform.yield_()

    # Fuse all sibling elementwise ops in scf.for loops.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    # Cleanup after tiling and fusion.
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize
    func = vectorize(mod, payload_func_name=payload_func_name)
    transform.apply_cse(func)
    canonicalize(func)

    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # bufferize
    mod = bufferize(mod)

    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    convert_to_gpu_launch(mod, payload_func_name=payload_func_name)

    func = get_payload_func(mod, func_name=payload_func_name)
    # set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    num_subgroups = layer_params["wg_rows"] // layer_params["sg_rows"]
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

    # Set layout attributes for xegpu.store_nd and xegpu.store_matrix ops.
    gpu_mod = match_and_split(mod, ops={"gpu.module"})[0]
    gpu_func = match(gpu_mod, ops={"gpu.func"})
    store_nd_ops = match(gpu_func, ops={"xegpu.store_nd"})
    store_matrix_ops = match(gpu_func, ops={"xegpu.store_matrix"})
    sg_layout = [layer_params["wg_rows"] // layer_params["sg_rows"], 1]
    sg_data = [layer_params["sg_rows"], layer_params["reduction_step_size"]]
    with lh_transform.foreach(store_nd_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()
    with lh_transform.foreach(store_matrix_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

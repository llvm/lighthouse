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
    1. Tiles the linalg loops for WG parallelism and reduction dimensions.
    2. Vectorize operations
    3. Bufferize tensors
    4. Convert to GPU dialect
    5. Lower to XeGPU operations
    6. Adds XeGPU layout attributes

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        params: ScheduleParameters object containing one dictionary with keys:
            - wg_tile: nD tile sizes for workgroup tiling
            - wg_subtile: nD tile sizes for generating a persistent WG subtile
                loop (optional)
            - sg_tile: nD tile sizes for subgroup tiling
            - reduction_tile: nD tile sizes for reduction dimension
            - subgroup_size: Size of subgroup (typically 16)

    The `wg_tile` and `sg_tile` tile sizes determine how the problem is
    partitioned across workgroups and subgroups. Only parallel dimensions can
    be tiled. The `reduction_tile` size determines how the reduction dimension
    is tiled. In both cases zero entries mean that a dimension is not tiled.

    For example, if the kernel takes a 4D input tensor with shape (128, 64,
    512, 512) and the reduction operator has iteration space ["parallel",
    "reduction", "parallel", "parallel"], then the following parameters define
    a valid tiling scheme:

        wg_tile = [0, 0, 256, 256]  # Tile the last two parallel dims
        sg_tile = [0, 0, 32, 32]    # Tile the last two parallel dims
        reduction_tile = [0, 32, 0, 0]  # Tile the reduction dimension

    Trailing zeros can be omitted from the tile size list, e.g., [0, 32] is
    equivalent to [0, 32, 0, 0].

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

    # TODO validate the dimensionality of the tile sizes and reduction dimensions
    wg_tile = layer_params["wg_tile"]
    wg_subtile = layer_params.get("wg_subtile")
    sg_tile = layer_params["sg_tile"]
    reduction_tile = layer_params["reduction_tile"]
    subgroup_size = layer_params["subgroup_size"]

    assert sum(wg_tile) > 0, "wg_tile must have at least one non-zero value"
    assert sum(sg_tile) > 0, "sg_tile must have at least one non-zero value"
    assert sum(reduction_tile) > 0, (
        "reduction_tile must have at least one non-zero value"
    )

    assert len(wg_tile) == len(sg_tile) == len(reduction_tile), (
        "wg_tile, sg_tile, and reduction_tile must have the same number of dimensions"
    )
    if wg_subtile:
        assert len(wg_subtile) == len(wg_tile), (
            "wg_subtile must have the same number of dimensions as wg_tile"
        )
    apply_wg_subtile = wg_subtile is not None and sum(wg_subtile) > 0

    ndims = len(wg_tile)
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
    generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
    leaf_generic = transform_ext.extract_handle(generic_ops, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf_generic,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    lh_transform.cleanup(func)

    if apply_wg_subtile:
        # Add a persistent loop to reduce wg tile size.
        generic_ops = structured.structured_match(anytype, func, ops=["linalg.generic"])
        leaf_generic = transform_ext.extract_handle(generic_ops, -1)
        lh_transform.tile(
            leaf_generic,
            tile_sizes=wg_subtile,
            fuse_producers=True,
            use_forall=False,
            apply_cleanup=False,
        )
        lh_transform.cleanup(func)

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

    def tile_and_fuse_reduction(reduction_op, tile_sizes):
        # Tile the reduction op.
        tiled_op, tile_loops, _ = lh_transform.tile(
            reduction_op,
            tile_sizes=tile_sizes,
            fuse_producers=False,
            use_forall=False,
            apply_cleanup=False,
        )
        fuse_elemwise_producers_to_loop(tiled_op, tile_loops[0])

    apply_reduction_tiling = sum(reduction_tile) > 0

    if apply_reduction_tiling:
        # Reduction dimension tiling.
        # 1. Tile the leaf elemwise linalg.generic op and fuse its elemwise
        #    linalg.generic producers into the resulting loop.
        # 2. Tile each reduction linalg.generic op (from last to first) and fuse its
        #    elemwise producers into the resulting loop.

        wg_loop = match_and_split(func, ops={"scf.forall"}, nhandles=1)[0]
        generic_ops = match(wg_loop, ops={"linalg.generic"})
        elemwise_ops = transform_ext.filter_elementwise(generic_ops)
        leaf_elemwise = transform_ext.extract_handle(elemwise_ops, -1)
        reduction_ops = transform_ext.filter_reduction_ops(generic_ops)

        # Tile trailing elemwise op first.
        tiled_elemwise, tile_loop = structured.TileUsingForOp(
            leaf_elemwise, sizes=reduction_tile
        ).results
        # Fuse all elemwise producers into the tiled leaf loop.
        elemwise_for = fuse_elemwise_producers_to_loop(tiled_elemwise, tile_loop)

        # Sink the loop's init tensor.extract_slice into the loop, dropping the
        # surrounding extract/insert_slice pair. This is required for clean
        # vectorization.
        elemwise_for = transform_ext.sink_extract_slice_into_loop(elemwise_for)

        # Tile and fuse the reduction loops in reverse order. After each fusion
        # step, DCE removes the dead untiled elementwise epilogue so it cannot
        # create a cross-loop use that breaks the next tile-fuse iteration. Note
        # that DCE does not invalidate the reduction loop handles as the tracking
        # listener only invalidates modified handles and the reduction loops are
        # alive and thus not removed.
        reduction_ops = transform_ext.reverse_handles(reduction_ops)
        with lh_transform.foreach(reduction_ops) as reduction_op:
            tile_and_fuse_reduction(reduction_op, reduction_tile)
            transform.apply_dce(wg_loop)
            transform.yield_()

        # Fuse all sibling elementwise ops in scf.for loops.
        func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")

    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()

    # Cleanup after tiling and fusion.
    lh_transform.cleanup(func)

    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # vectorize
    # Disable multi-reduction to contract patterns because xegpu lowering does
    # not currently support vector.contract reductions properly.
    func = vectorize(
        mod,
        payload_func_name=payload_func_name,
        disable_multi_reduction_to_contract_patterns=True,
    )

    # Convert math.fpowi to arith.mulf.
    func = get_payload_func(mod, func_name=payload_func_name)
    func = apply_registered_pass(func, "math-expand-ops")

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
    num_subgroups = 1
    _wg_tile = wg_subtile if apply_wg_subtile else wg_tile
    for i, (wg, red, sg) in enumerate(zip(_wg_tile, reduction_tile, sg_tile)):
        if wg > 0 and sg > 0:
            num_subgroups *= wg // sg
        if red > 0 and sg > 0:
            num_subgroups *= red // sg
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

    mod = convert_vector_to_xegpu(mod)
    lh_transform.cleanup(mod)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Set layout attributes for xegpu.store_nd and xegpu.store_matrix ops.
    gpu_mod = match_and_split(mod, ops={"gpu.module"})[0]
    gpu_func = match(gpu_mod, ops={"gpu.func"})
    store_nd_ops = match(gpu_func, ops={"xegpu.store_nd"})
    store_matrix_ops = match(gpu_func, ops={"xegpu.store_matrix"})
    sg_layout = [1] * ndims
    for i, (wg, red, sg) in enumerate(zip(_wg_tile, reduction_tile, sg_tile)):
        if red > 0 and sg > 0:
            sg_layout[i] = int(red // sg)
        if wg > 0 and sg > 0:
            sg_layout[i] = int(wg // sg)
    sg_data = [1] * ndims
    for i, (sg, red) in enumerate(zip(sg_tile, reduction_tile)):
        if sg > 0:
            sg_data[i] = sg
        elif red > 0:
            sg_data[i] = red
    if ndims == 4 and wg_tile[0] == 1 and sg_tile[0] == 0:
        # Drop the first entry as batch dim is extracted with a memref.subview.
        sg_layout = sg_layout[1:]
        sg_data = sg_data[1:]
    with lh_transform.foreach(store_nd_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()
    with lh_transform.foreach(store_matrix_ops) as store_op:
        xegpu.set_anchor_layout(store_op, sg_layout=sg_layout, sg_data=sg_data)
        transform.yield_()
    if apply_wg_subtile:
        # Set layout attributes for xegpu.load_nd op in the persistent wg subtile loop.
        load_nd_ops = match(gpu_func, ops={"xegpu.load_nd"})
        first_load_op = transform_ext.extract_handle(load_nd_ops, 0)
        xegpu.set_anchor_layout(first_load_op, sg_layout=sg_layout, sg_data=sg_data)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

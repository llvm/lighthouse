"""Generate MLIR transform schedule for XeGPU layer_norm operation."""

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
from lighthouse.schedule import schedule_boilerplate
from lighthouse.dialects.transform import transform_ext


def layer_norm_schedule(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> ir.Module:
    """
    Generate transform schedule for layer_norm operation.

    The schedule performs the following transformations:
    1. Tile the outer parallel dimension (rows) using forall
    2. Tile the inner reductions (mean / variance) using for
    3. Vectorize operations
    4. Bufferize tensors
    5. Convert to GPU dialect
    6. Lower to XeGPU operations

    Args:
        stop_at_stage: Optional stage name to stop early (for debugging)
        parameters: Dictionary with scheduling parameters:
            - wg_rows: Number of rows per workgroup
            - sg_rows: Number of rows per subgroup
            - subgroup_size: Size of subgroup
            - sizes: Tuple with the sizes of the input tensors (e.g. (M, N))
            - reduction_step_size: Step size for tiling the reduction loops
    """
    assert parameters is not None, "Schedule parameters must be provided"

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func = match(named_seq.bodyTarget, ops={"func.func"})
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )

        try:
            bundle_xegpu_layer_norm_schedule(
                payload_mod,
                parameters=parameters,
                stop_at_stage=stop_at_stage,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()

    return schedule


def bundle_xegpu_layer_norm_schedule(
    mod: ir.Value,
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value:
    """Schedule for lowering layer_norm payload to xegpu wg level.

    The payload (see ``generate_gpu_layer_norm_payload``) consists of:
      - linalg.fill (init mean accumulator)
      - linalg.generic (mean reduction)
      - linalg.fill (init var accumulator)
      - linalg.generic (var reduction)
      - linalg.generic (final normalize: elementwise)
    """

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()
    reduction_step_size = parameters["reduction_step_size"]

    # Get the payload function by anchoring on the last linalg.generic
    # (the elementwise normalize op, which is the only op with 2 parallel iterators).
    all_generics = match(mod, ops={"linalg.generic"})
    # Split: 3 generics in total (mean reduction, var reduction, normalize).
    gen_ops = transform.split_handle((anytype,) * 3, all_generics)
    mean_reduction = gen_ops[0]
    var_reduction = gen_ops[1]
    normalize_op = gen_ops[2]

    tiled_op, forall_op = structured.structured_tile_using_forall(
        anytype,
        anytype,
        normalize_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(parameters["wg_rows"],),
    )

    # Fuse the two reductions into the forall.
    _, forall_op = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=var_reduction,
        containing_op=forall_op,
    )
    _, forall_op = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=mean_reduction,
        containing_op=forall_op,
    )

    func = transform.get_parent_op(
        anytype,
        forall_op,
        op_name="func.func",
        deduplicate=True,
    )

    # Fuse the (M,)-sized accumulator init fills (mean_acc, var_acc) into the
    # forall as well. Fusing the reductions only slices the fill *results*
    # inside the loop; the full-size fills themselves stay at function scope.
    # If left outside, bufferization turns them into a function-scope
    # memref<Mxf32> that is passed into gpu.launch as a host pointer, which the
    # device then dereferences -> GPU page fault. Privatizing them into the
    # forall makes each a per-workgroup (wg_rows,) init inside the kernel.
    fill_ops = match(func, ops={"linalg.fill"})
    _, forall_op = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=fill_ops,
        containing_op=forall_op,
    )

    # Drop the dead originals of the reductions (canonicalize removes them)
    # before re-matching, otherwise we'd see 5+ generics instead of 3.
    transform.apply_cse(func)
    canonicalize(func)

    # Re-match the linalg.generic ops after fusion: 3 generics remain
    # (mean reduction, var reduction, normalize).
    linalg_ops = match_and_split(func, ops={"linalg.generic"}, nhandles=3)
    mean_reduction = linalg_ops[0]
    var_reduction = linalg_ops[1]
    normalize_op = linalg_ops[2]

    # Tile the elementwise normalize along its inner (column) dim.
    _, normalize_loop = structured.TileUsingForOp(
        normalize_op, sizes=[0, reduction_step_size]
    ).results

    # Tile the variance reduction along its reduction dim.
    structured.structured_tile_reduction_using_for(
        [anytype],
        anytype,
        anytype,
        anytype,
        target=var_reduction,
        tile_sizes=[0, reduction_step_size],
    )

    # Tile the mean reduction along its reduction dim.
    structured.structured_tile_reduction_using_for(
        [anytype],
        anytype,
        anytype,
        anytype,
        target=mean_reduction,
        tile_sizes=[0, reduction_step_size],
    )

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

    transform.apply_cse(mod)
    canonicalize(mod)

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # Set layout attributes for xegpu.store_nd and xegpu.store_matrix ops.
    sg_layout = [parameters["sg_rows"], 1]
    sg_data = [parameters["sg_rows"], parameters["reduction_step_size"]]
    store_nd_ops = match(gpu_func, ops={"xegpu.store_nd"})
    xegpu.set_anchor_layout(store_nd_ops, sg_layout=sg_layout, sg_data=sg_data)
    store_matrix_ops = match(gpu_func, ops={"xegpu.store_matrix"})
    xegpu.set_anchor_layout(store_matrix_ops, sg_layout=sg_layout, sg_data=sg_data)

    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()

    return mod

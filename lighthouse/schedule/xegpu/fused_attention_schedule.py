"""Generate MLIR transform schedule for XeGPU fused attention operation."""

from typing import Optional

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform.vector import (
    apply_patterns_vector_cast_away_vector_leading_one_dim,
)

from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
    apply_registered_pass,
)
from lighthouse.schedule import schedule_boilerplate


def fused_attention_schedule(
    stop_at_stage: Optional[str] = None,
    parameters: Optional[dict] = None,
) -> ir.Module:
    """
    Generate transform schedule for attention kernel.

    The schedule performs the following transformations:
    1. Tile the fuse the strandard attention computation along parallel dims
    2. Vectorize operations
    3. Bufferize tensors
    4. Perform the fused attention optimization for the innermost computation
    5. Convert to GPU dialect
    6. Lower to XeGPU operations

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


def tile_and_fuse_parallel_dim(
    func: ir.Value[transform.AnyOpType],
    anytype: transform.AnyOpType,
    parameters: dict,
) -> None:
    """Outer tiling: partition the whole computation along the parallel batch
    dimension (batch = Z * H) into an scf.forall over batch tiles.

    The final division op produces the output; its iteration space is
    (batch, n_ctx, n_head) and is fully parallel, so tiling it along dim 0 with
    tile_using_forall creates an scf.forall over batch tiles. Every producer
    (the two matmuls, the two reductions, the scale multiply, the transpose and
    the initialization fills) is then fused into that scf.forall so each
    workgroup computes attention for its own slice of batch elements.
    """
    # Match the structured ops that remain after generalization + fusion. Order
    # of the 5 linalg.generic ops in the payload:
    #   0: QK^T matmul   1: row max   2: row sum   3: P*V matmul   4: div
    qkt, max_reduction, sum_reduction, pv_matmul, div_op = match_and_split(
        func, ops={"linalg.generic"}, nhandles=5
    )
    # The scale multiply (linalg.mul) and the transpose still sit between the
    # QK^T matmul and the softmax reductions; grab them so we can pull the whole
    # QK^T producer chain into the scf.forall too.
    scale_mul = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]
    transpose_op = match_and_split(func, ops={"linalg.transpose"}, nhandles=1)[0]

    wg_tile_size = parameters.get("wg_rows", 128)
    _, forall_loop = structured.structured_tile_using_forall(
        anytype,
        anytype,
        div_op,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_tile_size, 0),
    )
    # Fuse consumers-to-producers. Both the sum reduction and the P*V matmul feed
    # the final division; whichever is fused last is cloned closest to the
    # division. Fuse the sum reduction before the P*V matmul so the in-loop order
    # stays (row max, row sum, P*V). The inner reduction fusion needs the sum to
    # precede the P*V matmul: fusing the sum into the tiled max loop requires
    # every *other* user of the max result (here the P*V matmul) to post-dominate
    # the sum, which only holds if the sum comes first.
    for producer in [
        pv_matmul,
        sum_reduction,
        max_reduction,
        scale_mul,
        qkt,
        transpose_op,
    ]:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=producer, containing_op=forall_loop
        )

    # Also fuse the linalg.fill ops that initialize the matmul/reduction outputs
    # into the scf.forall. Besides keeping each workgroup self-contained, this
    # rewrites the reduction inits from `tensor.extract_slice(fill)` (outside the
    # loop) into `fill(extract_slice(empty))` (inside the loop), so the
    # dependant-reduction fusion's zero-init legality check can see through them.
    # There are 5 fills: QK^T out, scale tensor, max init, sum init, P*V out.
    fill_ops = match_and_split(func, ops={"linalg.fill"}, nhandles=5)
    for fill_op in fill_ops:
        _, forall_loop = structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=fill_op, containing_op=forall_loop
        )
    transform.apply_cse(func)
    canonicalize(func)


def tile_and_fuse_reduction_dim(
    func: ir.Value[transform.AnyOpType],
    anytype: transform.AnyOpType,
    parameters: dict,
) -> None:
    """Build the fused inner reduction loop (online softmax) inside the batch
    scf.forall.

    Within each batch tile, tile the row-max reduction along the K/V sequence
    (reduction) dimension and fuse the dependant sum reduction and the P*V
    matmul into that loop, then pull the QK^T producer chain in as well. The
    result is a single scf.for reduction loop implementing online softmax.
    """
    # Re-match the ops now living inside the scf.forall. Fusing producers with
    # multiple uses can duplicate them, so threading handles through the outer
    # fusion is unreliable; matching by op type inside the loop is robust. After
    # outer tiling the 5 linalg.generic ops appear in this order:
    #   0: QK^T matmul   1: row max   2: row sum   3: P*V matmul   4: div
    qkt, max_reduction, sum_reduction, pv_matmul, _div = match_and_split(
        func, ops={"linalg.generic"}, nhandles=5
    )
    scale_mul = match_and_split(func, ops={"linalg.mul"}, nhandles=1)[0]
    transpose_op = match_and_split(func, ops={"linalg.transpose"}, nhandles=1)[0]

    # Grab the QK^T-out and scale fills so step 6 can sink them into the scf.for
    # alongside the QK^T producer chain. In program order after outer tiling the
    # 5 fills are: 0 QK^T out, 1 scale tensor, 2 row-max init, 3 P*V out,
    # 4 row-sum init. Only fills 0 and 1 belong to the transient QK^T block; the
    # other three are loop-carried softmax accumulators and stay in the forall.
    # These two handles are not touched by the reduction tiling in steps 3-5
    # (they feed qkt/scale_mul, which are only fused in step 6), so they survive
    # just like the scale_mul/qkt handles above.
    qkt_out_fill, scale_fill, _, _, _ = match_and_split(
        func, ops={"linalg.fill"}, nhandles=5
    )

    reduction_step_size = parameters.get("inner_loop_tile_size", 64)

    # Step 3: Tile the producer reduction (row max) along its reduction dim (the
    # K/V sequence length, the last iteration dim of the 3D iteration space) and
    # annotate the resulting loop so the fusion op recognises it as a tiled
    # reduction.
    _, max_loop = structured.TileUsingForOp(
        max_reduction, sizes=[0, 0, reduction_step_size]
    ).results
    transform.annotate(max_loop, "__reduction_loop__")

    # Step 4: Fuse the dependant sum reduction into the tiled max loop.
    fused_loop = structured.structured_tile_and_fuse_dependant_reduction_ops(
        anytype, max_loop, sum_reduction
    )
    transform.annotate(fused_loop, "__reduction_loop__")

    # Step 5: Fuse the dependant P*V matmul into the fused loop.
    fused_loop = structured.structured_tile_and_fuse_dependant_reduction_ops(
        anytype, fused_loop, pv_matmul
    )
    transform.apply_cse(func)
    canonicalize(func)

    # Step 6: Fuse the QK^T producer chain into the scf.for reduction loop. The
    # loop slices the scaled QK^T tensor, so fuse from the closest producer
    # outward: the scale multiply, then the QK^T matmul, then the K transpose.
    # The QK^T-out and scale fills are pulled in right after the op that consumes
    # them (qkt reads qkt_out_fill as its init, scale_mul reads scale_fill as an
    # input) so they become per-iteration 1x128x64 inits instead of full-tile
    # 1x128x4096 buffers. After vectorization each collapses to its splat
    # constant, eliminating the two 1x128x4096 scratch allocs entirely.
    for producer in [scale_mul, scale_fill, qkt, qkt_out_fill, transpose_op]:
        _, fused_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=producer,
            containing_op=fused_loop,
        )
    transform.apply_cse(func)
    canonicalize(func)


def bundle_xegpu_fused_attention_schedule(
    mod: ir.Value[transform.AnyOpType],
    parameters: dict,
    stop_at_stage: str = "",
) -> ir.Value[transform.AnyOpType]:
    """Schedule for lowering attention payload to xegpu wg level."""

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    anytype = transform.AnyOpType.get()

    # The payload emits attention with softmax already in decomposed form and the
    # final division deferred until after the P*V matmul (flash-attention style):
    #   1. QK^T           (batch_matmul)
    #   2. scale          (linalg.mul)
    #   3. row max        (generic reduction)
    #   4. P = exp(x-max) (generic elementwise, the softmax numerator)
    #   5. row sum        (generic reduction over P)
    #   6. P*V            (batch_matmul)
    #   7. out = PV / sum (generic elementwise, the deferred division)
    #
    # The computation is tiled at two levels: an outer scf.forall over the
    # parallel batch dimension (tile_and_fuse_parallel_dim), and an inner
    # scf.for over the softmax/P*V reduction dimension implementing online
    # softmax via the dependant-reduction fusion (tile_and_fuse_reduction_dim).

    # Match both matmuls (QK^T and P*V) and scope the rest of the schedule to
    # the payload function that contains them.
    matmul_ops = match(mod, ops={"linalg.batch_matmul"})
    func = transform.get_parent_op(
        anytype,
        matmul_ops,
        op_name="func.func",
        deduplicate=True,
    )
    transform.print_(target=func, name="initial payload function")

    # Step 1: Generalize both matmuls (QK^T and P*V) into linalg.generic ops so
    # the reduction-fusion machinery can operate on them uniformly.
    structured.structured_generalize(anytype, matmul_ops)
    transform.apply_cse(func)
    canonicalize(func)

    # Step 2: Fuse elementwise ops into their producers/consumers. This folds the
    # softmax numerator P = exp(x - max) into both the sum reduction and the P*V
    # matmul, leaving the two matmuls, the two reductions and the final division
    # as the remaining structured ops.
    func = apply_registered_pass(func, "linalg-fuse-elementwise-ops")
    transform.apply_cse(func)
    canonicalize(func)

    # Outer tiling: partition the whole computation along the parallel batch
    # dimension so each workgroup computes attention for its own batch slice.
    tile_and_fuse_parallel_dim(func, anytype, parameters)
    transform.print_(target=func, name="after outer tiling and fusion")

    if stop_at_stage == "outer-tiled":
        raise PipelineInterrupt()

    # Inner tiling: within each batch tile, build a single fused reduction loop
    # over the softmax / P*V reduction dimension. Fusing the linalg.fill ops into
    # the scf.forall above rewrote the reduction inits into in-loop
    # `linalg.fill`s (rather than `tensor.extract_slice` of an outside fill), so
    # the dependant-reduction fusion's zero-init legality check now succeeds.
    tile_and_fuse_reduction_dim(func, anytype, parameters)
    # Drop the batch unit extent (tile size 1) from the linalg ops in the loop
    # nest before vectorizing. Tiling the batch dim by 1 left every op with a
    # leading 1x... shape; folding it here (via rank-reducing slices) means
    # vectorization emits 128x64 / 128 vectors directly instead of 1x128x64
    # vectors wrapped in shape_casts, and bufferization allocates unit-dim-free
    # memrefs (e.g. memref<128xf16> instead of memref<1x128xf16>).
    # The fold_unit_extent_dims patterns only rewrite linalg.generic ops, so
    # first generalize the remaining named/category ops in the loop nest (the K
    # transpose, the scale linalg.mul and the div/mul linalg.elementwise ops)
    # into generics. Without this the unit dim survives on exactly those ops.
    func = apply_registered_pass(
        func,
        "linalg-morph-ops",
        options={"category-to-generic": True},
    )
    transform.apply_cse(func)
    canonicalize(func)
    transform.print_(target=func, name="after generalizing")

    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_reshapes()
    transform.apply_cse(func)
    canonicalize(func)
    transform.print_(target=func, name="after inner tiling and fusion")

    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    # Vectorize the fused loop nest: rewrite the remaining linalg ops (the tiled
    # matmuls, reductions and elementwise ops inside the scf.for) into vector
    # ops.
    func = structured.VectorizeChildrenAndApplyPatternsOp(
        func,
        fold_type_extensions_into_contract=True,
    ).result
    transform.apply_cse(func)
    canonicalize(func)

    # Try to remove any unit dimensions that may have been introduced due to tiling (e.g. batch dim of 1)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        apply_patterns_vector_cast_away_vector_leading_one_dim()

    transform.print_(target=func, name="after vectorization")
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    # Bufferize: convert tensors to memrefs across function boundaries with an
    # identity layout map, then fold memref.subviews into the vector transfer
    # ops.
    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    identity_layout = LayoutMapOption.IdentityLayoutMap
    mod = transform_bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=False,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=identity_layout,
    ).result
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    # Promote small memref.allocs (the per-workgroup scratch buffers) to the
    # stack in the payload function.
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(
        func,
        "promote-buffers-to-stack",
        options={
            "max-alloc-size-in-bytes": "16384",
            "max-rank-of-allocated-memref": "3",
        },
    )

    transform.print_(target=func, name="after bufferization")
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # Convert forall to parallel
    wg_loops = match_and_split(mod, ops={"scf.forall"})
    for wg_loop in wg_loops:
        wg_loop = loop.loop_forall_to_parallel([anytype], wg_loop)
    func = transform.get_parent_op(anytype, wg_loop)

    # Convert scf.parallel to gpu.launch
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # Set the number of threads for the gpu.launch operation
    launch_op = match_and_split(func, ops={"gpu.launch"})
    wg_rows = parameters["wg_rows"]
    sg_rows = parameters["sg_rows"]
    subgroup_size = parameters["subgroup_size"]
    num_subgroups = wg_rows // sg_rows
    num_threads = num_subgroups * subgroup_size
    xegpu.set_gpu_launch_threads(launch_op[0], threads=[num_threads, 1, 1])

    # Outline gpu func
    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)

    transform.print_(target=mod, name="after gpu kernel outlining")
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    # Set xevm target
    mod = apply_registered_pass(
        mod,
        "xevm-attach-target",
        options={"O": "3", "chip": "bmg"},
    )

    # Convert vectot to xegpu
    gpu_mod_ops = match_and_split(mod, ops={"gpu.module"})
    for gpu_mod in gpu_mod_ops:
        gpu_func = match(gpu_mod, ops={"gpu.func"})
        allocas = match(gpu_func, ops={"memref.alloca"})
        transform_ext.update_address_space(allocas, address_space=3)
        gpu_func = apply_registered_pass(gpu_func, "convert-vector-to-xegpu")
        transform.apply_cse(gpu_func)
        gpu_func = apply_registered_pass(gpu_func, "loop-invariant-code-motion")

    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    return mod

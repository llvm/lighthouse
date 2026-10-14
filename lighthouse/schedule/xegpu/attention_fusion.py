"""Shared tensor-level online-attention fusion transforms."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import match, apply_registered_pass
from lighthouse.dialects.transform import transform_ext


def transform_to_flash_attention(
    anytype, func, reduction_tile, n_parallel
) -> ir.Operation:
    """Fold the attention chain into the tiled row-max loop.

    After generalization: ``s = Q@K^T * scale``, ``m = max(s)``,
    ``p = exp(s-m)``, ``o = p@V``, ``l = sum(p)``, and ``out = o/l``.
    Fuse the row sum and P@V separately, each with its own online correction.
    Assumes finite V and finite or -inf scores; the output normalization must
    preserve the fully-masked-row guard. All--inf row masks are rewritten using
    the running/final maximum before sinking score producers.
    """
    # Match the necessary op handles. Number of reduction and elementwise loops
    # can vary due to possible prologue/epilogue and P mask generation.

    # Match sum reduction as the final reduction op.
    reductions = transform_ext.filter_reduction_ops(
        lh_transform.match_op(func, "linalg.generic")
    )
    sum_op = transform_ext.extract_handle(reductions, -1)

    # Match max reduction op. Assumes there's only one arith.max* op.
    arith_max_op = transform_ext.extract_handle(
        lh_transform.match_op(func, ["arith.maximumf", "arith.maxnumf"]),
        0,
        silenceable=True,
    )
    max_op = transform.get_parent_op(anytype, arith_max_op, op_name="linalg.generic")
    # Keep track of the producers of the max reduction.
    max_producers = transform_ext.trace_producers(max_op)
    max_producers = transform_ext.filter_by_name(
        max_producers,
        ["linalg.generic", "linalg.transpose"],
    )

    # Match P@V contraction, assuming it's the second contraction op.
    linalg_ops = structured.structured_match(
        anytype, func, ops=["linalg.generic", "linalg.batch_matmul"]
    )
    contraction_ops = transform_ext.filter_contraction_ops(linalg_ops)
    pv_op = transform_ext.extract_handle(contraction_ops, 1, silenceable=True)

    # Match P op as the loop with exp.
    exp_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "math.exp"), 0, silenceable=True
    )
    p_op = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")

    # Tile max loop, only the final reduction dimension.
    sizes = [0] * n_parallel + [reduction_tile]
    _, [reduction_loop], _ = lh_transform.tile(max_op, tile_sizes=sizes)
    transform.annotate(reduction_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

    # Fuse the row sum first. P@V still needs the original exp term.
    reduction_loop = transform_ext.fuse_dependent_reduction_ops(
        p_op, sum_op, reduction_loop
    )

    # Reacquire the exp term after the first fusion consumes its handle.
    exp_op = transform_ext.extract_handle(
        lh_transform.match_op(func, "math.exp"), -1, silenceable=True
    )
    p_op = transform.get_parent_op(anytype, exp_op, op_name="linalg.generic")
    # Fuse P@V contraction into the reduction loop.
    reduction_loop = transform_ext.fuse_dependent_reduction_ops(
        p_op, pv_op, reduction_loop
    )
    reduction_loop = transform_ext.rewrite_attention_row_mask(reduction_loop)
    transform.apply_dce(func)
    transform.apply_cse(func)

    def fuse_fill_producers_to_loop(target, parent_loop):
        """Fuses all elementwise producer ops of `target` into `parent_loop`."""
        producers = transform_ext.trace_producers(target)
        elemwise_producers = transform_ext.filter_elementwise(producers)
        elemwise_producers = transform_ext.filter_by_name(
            elemwise_producers,
            "linalg.fill",
        )
        _, fused_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=elemwise_producers,
            containing_op=parent_loop,
        )
        return fused_loop

    # Fuse all max producers into the reduction loop.
    with lh_transform.foreach(max_producers) as producer:
        tiled_op, reduction_loop = structured.structured_fuse_into_containing_op(
            anytype,
            anytype,
            producer_op=producer,
            containing_op=reduction_loop,
        )
        transform.apply_dce(func)
        fuse_fill_producers_to_loop(tiled_op, reduction_loop)
        transform.yield_()

    transform.apply_cse(func)
    func = apply_registered_pass(func, "remove-dead-values")
    lh_transform.cleanup(func)

    # Drop the unit batch dim after fusion so XeGPU sees rank-2 tiles. Generalize
    # named ops for the linalg fold; tensor patterns then compose leftover slices.
    named_ops = lh_transform.match_op(
        func, ["linalg.transpose", "linalg.fill", "linalg.elementwise"]
    )
    structured.structured_generalize(anytype, named_ops)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        structured.apply_patterns_linalg_fold_unit_extent_dims_via_slices()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)
    with ir.InsertionPoint(transform.apply_patterns(func).patterns):
        tensor.apply_patterns_tensor_merge_consecutive_insert_extract_slice()
        tensor.apply_patterns_tensor_drop_redundant_insert_slice_rank_expansion()
        tensor.apply_patterns_tensor_fold_tensor_subset_ops()
        transform.apply_patterns_canonicalization()
    transform.apply_cse(func)
    return func


def annotate_fastmath_flags(func: ir.Value[transform.AnyOpType]) -> None:
    """Mark online rescale quotients for LLVM's MathToXeVM simplification.

    Preserve any flags already chosen by the payload. `afn` on the exps
    selects native math; `reassoc,arcp` on each quotient permits exp(a-b).
    """
    no_fastmath = ir.Attribute.parse("#arith.fastmath<none>")
    fast = transform.param_constant(
        transform.AnyParamType.get(), ir.Attribute.parse("#arith.fastmath<fast>")
    )
    unmarked = match(func, ops=["math.exp"], op_attrs={"fastmath": no_fastmath})
    transform.annotate(unmarked, "fastmath", param=fast)

    quotient_flags = transform.param_constant(
        transform.AnyParamType.get(),
        ir.Attribute.parse("#arith.fastmath<reassoc,arcp>"),
    )
    reduction_loop = match(
        func,
        ops=["scf.for"],
        op_attrs={transform_ext.REDUCTION_LOOP_ATTR_NAME: ir.UnitAttr.get()},
    )
    correction_divs = match(
        reduction_loop,
        ops=["arith.divf"],
        op_attrs={"fastmath": no_fastmath},
    )
    transform.annotate(correction_divs, "fastmath", param=quotient_flags)

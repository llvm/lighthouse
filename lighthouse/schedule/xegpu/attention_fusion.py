"""Shared tensor-level online-attention fusion transforms."""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured, tensor
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import canonicalize, match
from lighthouse.dialects.transform import transform_ext


def derive_flash_attention(anytype, func, reduction_tile, n_parallel):
    """Fold the attention chain into the tiled row-max loop.

    After generalization: ``s = Q@K^T * scale``, ``m = max(s)``,
    ``p = exp(s-m)``, ``o = p@V``, ``l = sum(p)``, and ``out = o/l``.
    Fuse the row sum and P@V separately, each with its own online correction.
    """
    # Four reduction generics remain after WG tiling, in this order.
    reductions = transform_ext.filter_reduction_ops(match(func, ops={"linalg.generic"}))
    qk_op, max_op, pv_op, sum_op = transform.split_handle([anytype] * 4, reductions)

    # Follow operands to the exp term and K transpose.
    prod = transform.get_producer_of_operand
    p_op = prod(anytype, sum_op, operand_number=0)
    scale_mul_op = prod(anytype, max_op, operand_number=0)
    transpose_op = prod(anytype, qk_op, operand_number=1)

    # Tile only the final reduction axis. The parameterized schedule uses a
    # static size; the modular schedule supplies a transform parameter.
    sizes = [0] * n_parallel + [reduction_tile]
    if isinstance(reduction_tile, int):
        _, reduction_loop = structured.structured_tile_using_for(
            anytype,
            [anytype],
            max_op,
            dynamic_sizes=[],
            interchange=[],
            static_sizes=sizes,
            scalable_sizes=[False] * len(sizes),
        )
    else:
        _, [reduction_loop], _ = lh_transform.tile(max_op, tile_sizes=sizes)
    transform.annotate(reduction_loop, transform_ext.REDUCTION_LOOP_ATTR_NAME)

    # Fuse the row sum first. P@V still needs the original exp term.
    reduction_loop = transform_ext.fuse_dependent_reduction_ops(
        p_op, sum_op, reduction_loop
    )

    # Reacquire the exp term after the first fusion consumes its handle.
    p_op = prod(anytype, pv_op, operand_number=0)
    reduction_loop = transform_ext.fuse_dependent_reduction_ops(
        p_op, pv_op, reduction_loop
    )
    transform.apply_cse(func)

    # Sink score producers so only a score tile stays live.
    _, reduction_loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, scale_mul_op, reduction_loop
    )
    tiled_qk, reduction_loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, qk_op, reduction_loop
    )
    _, reduction_loop = structured.structured_fuse_into_containing_op(
        anytype, anytype, transpose_op, reduction_loop
    )

    # Sink the Q@K^T fill through its in-loop destination slice. Running
    # accumulator fills remain outside the loop.
    fill_slice = prod(anytype, tiled_qk, operand_number=2)
    fill_op = prod(anytype, fill_slice, operand_number=0)
    _, reduction_loop = structured.structured_fuse_into_containing_op(
        anytype,
        anytype,
        producer_op=fill_op,
        containing_op=reduction_loop,
    )

    transform.apply_cse(func)
    canonicalize(func)

    # Drop the unit batch dim after fusion so XeGPU sees rank-2 tiles. Generalize
    # named ops for the linalg fold; tensor patterns then compose leftover slices.
    named_ops = match(
        func, ops={"linalg.transpose", "linalg.fill", "linalg.elementwise"}
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


def annotate_fastmath_flags(func: ir.Value[transform.AnyOpType]) -> None:
    """Mark online rescale quotients for LLVM's MathToXeVM simplification.

    Preserve any flags already chosen by the payload. `afn` on the exps
    selects native math; `reassoc,arcp` on each quotient permits exp(a-b).
    """
    anytype = transform.AnyOpType.get()
    no_fastmath = ir.Attribute.parse("#arith.fastmath<none>")
    fast = transform.param_constant(
        transform.AnyParamType.get(), ir.Attribute.parse("#arith.fastmath<fast>")
    )
    for op_name in ("math.exp", "arith.subf"):
        unmarked = structured.structured_match(
            anytype, func, ops=[op_name], op_attrs={"fastmath": no_fastmath}
        )
        transform.annotate(unmarked, "fastmath", param=fast)

    quotient_flags = transform.param_constant(
        transform.AnyParamType.get(),
        ir.Attribute.parse("#arith.fastmath<reassoc,arcp>"),
    )
    reduction_loop = structured.structured_match(
        anytype,
        func,
        ops=["scf.for"],
        op_attrs={transform_ext.REDUCTION_LOOP_ATTR_NAME: ir.UnitAttr.get()},
    )
    correction_divs = structured.structured_match(
        anytype,
        reduction_loop,
        ops=["arith.divf"],
        op_attrs={"fastmath": no_fastmath},
    )
    transform.annotate(correction_divs, "fastmath", param=quotient_flags)

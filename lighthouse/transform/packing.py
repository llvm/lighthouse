from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def block_pack_matmuls(
    target: ir.Operation | ir.Value,
    block_factors: list[int],
    lhs_transpose_outer_block: bool = False,
    lhs_transpose_inner_block: bool = False,
    rhs_transpose_outer_block: bool = True,
    rhs_transpose_inner_block: bool = True,
):
    """
    Block pack all linalg matmuls in target's nested functions.

    Args:
        target: Handle to matcher's target
        block_factors: Block sizes (mb, nb, kb)
        lhs_transpose_outer_block: A matrix MB x KB => KB x MB
        lhs_transpose_inner_block: A matrix mb x kb => kb x mb
        rhs_transpose_outer_block: B matrix KB x NB => NB x KB
        rhs_transpose_inner_block: B matrix kb x nb => nb x kb
    """
    assert len(block_factors) == 3, (
        f"Expected 3 block factors but got {len(block_factors)}"
    )
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_registered_pass(
        transform.any_op_t(),
        func,
        "linalg-block-pack-matmul",
        options={
            "block-factors": block_factors,
            "lhs-transpose-outer-blocks": lhs_transpose_outer_block,
            "lhs-transpose-inner-blocks": lhs_transpose_inner_block,
            "rhs-transpose-outer-blocks": rhs_transpose_outer_block,
            "rhs-transpose-inner-blocks": rhs_transpose_inner_block,
        },
    )


def pack_propagation(target: ir.Operation | ir.Value):
    """
    Apply pack propagation patterns to the target.

    Args:
        target: Handle to target
    """
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        structured.apply_patterns_linalg_data_layout_propagation(poison_padding=True)
        structured.apply_patterns_linalg_extract_slice_sinking()
        structured.apply_patterns_linalg_fold_pack_unpack_into_empty()
        structured.apply_patterns_tensor_fold_into_pack_and_unpack()
        transform.apply_patterns_canonicalization()

from mlir import ir
from mlir.dialects import transform

from .builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def block_pack_matmuls(
    block_factors: tuple[int, int, int],
    lhs_transpose_outer_block: bool = False,
    lhs_transpose_inner_block: bool = False,
    rhs_transpose_outer_block: bool = True,
    rhs_transpose_inner_block: bool = True,
) -> ir.Module:
    """
    Block pack all matmuls.

    Pack a matmul operation into blocked layout with two levels of subdivision:
    - major 2D blocks - outer dimensions, consist of minor blocks
    - minor 2D blocks - inner dimensions, consist of scalar elements

    A 2D matmul MxNxK gets reshaped into blocked 4D representation
    as: [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb]
    where the (MB, NB, KB) dimensions represent the major blocks,
    and the (mb, nb, kb) are the minor blocks of their respective
    original 2D dimensions (M, N, K).

    Args:
        block_factors: Block sizes (mb, nb, kb)
        lhs_transpose_outer_block: A matrix MB x KB => KB x MB
        lhs_transpose_inner_block: A matrix mb x kb => kb x mb
        rhs_transpose_outer_block: B matrix KB x NB => NB x KB
        rhs_transpose_inner_block: B matrix kb x nb => nb x kb
    Returns:
        Schedule
    """
    if len(block_factors) != 3:
        raise ValueError(f"Expected 3 block factors but got {len(block_factors)}")

    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        transform.apply_registered_pass(
            transform.any_op_t(),
            ops,
            "linalg-block-pack-matmul",
            options={
                "block-factors": block_factors,
                "lhs-transpose-outer-blocks": lhs_transpose_outer_block,
                "lhs-transpose-inner-blocks": lhs_transpose_inner_block,
                "rhs-transpose-outer-blocks": rhs_transpose_outer_block,
                "rhs-transpose-inner-blocks": rhs_transpose_inner_block,
            },
        )
        lh_transform.pack_propagation(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

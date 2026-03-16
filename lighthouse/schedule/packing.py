from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
import lighthouse.transform as lh_transform


def pack_matmuls(
    block_factors: list[int],
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
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        ops = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        lh_transform.block_pack_matmuls(
            ops,
            block_factors=block_factors,
            lhs_transpose_outer_block=lhs_transpose_outer_block,
            lhs_transpose_inner_block=lhs_transpose_inner_block,
            rhs_transpose_outer_block=rhs_transpose_outer_block,
            rhs_transpose_inner_block=rhs_transpose_inner_block,
        )
        lh_transform.pack_propagation(named_seq.bodyTarget)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

from mlir import ir
from mlir.dialects import transform

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import block_pack_matmuls
from lighthouse.transform import pack_propagation
from lighthouse.transform import cleanup


def pack_matmuls(
    block_factors: list[int],
    lhs_transpose_outer_block: bool = False,
    lhs_transpose_inner_block: bool = False,
    rhs_transpose_outer_block: bool = True,
    rhs_transpose_inner_block: bool = True,
) -> ir.Module:
    """
    Pack all matmuls.

    Args:
        block_factors: Block sizes (mb, nb, kb)
        lhs_transpose_outer_block: A matrix MB x KB => KB x MB
        lhs_transpose_inner_block: A matrix mb x kb => kb x mb
        rhs_transpose_outer_block: B matrix KB x NB => NB x KB
        rhs_transpose_inner_block: B matrix kb x nb => nb x kb
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        block_pack_matmuls(
            named_seq.bodyTarget,
            block_factors=block_factors,
            lhs_transpose_outer_block=lhs_transpose_outer_block,
            lhs_transpose_inner_block=lhs_transpose_inner_block,
            rhs_transpose_outer_block=rhs_transpose_outer_block,
            rhs_transpose_inner_block=rhs_transpose_inner_block,
        )
        pack_propagation(named_seq.bodyTarget)
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

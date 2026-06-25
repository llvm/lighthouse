from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def amx_move_offsets() -> ir.Module:
    """
    Moves non-constant indicies from AMX input reads to subviews.
    This IR shapes helps AMX pattern to identify correct sub-graphs.

    Returns:
        Schedule module
    """
    with schedule_boilerplate() as (sched, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, "vector.transfer_read")
        transform_ext.move_offsets_to_subview(ops)
        func = lh_transform.match_op(named_seq.bodyTarget, "func.func")
        transform.apply_cse(func)
        transform.yield_()
    return sched

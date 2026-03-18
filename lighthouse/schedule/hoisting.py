from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform.structured import MatchInterfaceEnum

from .builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def hoist_loops(
    target_op: str
    | list[str]
    | MatchInterfaceEnum = MatchInterfaceEnum.LoopLikeInterface,
) -> ir.Module:
    """
    Apply loop hoisting to all matching ops.

    Args:
        target_op: Ops to be matched
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
        lh_transform.loop_hoisting(ops)
        lh_transform.cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

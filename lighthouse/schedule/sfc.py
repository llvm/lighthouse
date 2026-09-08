from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule.builders import schedule_boilerplate
import lighthouse.transform as lh_transform


def remap(
    target_op: str | list[str] | None = None,
) -> ir.Module:
    """
    Remaps `target_op` parent loop iteration space using
    the space-filling curve strategy.

    Args:
        target_op: Op(s) to consider. Defaults to all linalg ops.

    Returns:
        Schedule
    """
    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
        transform_ext.sfc_remap_forall(ops)
        transform.yield_()
    return schedule

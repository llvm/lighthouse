from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse.dialects.transform import transform_ext
import lighthouse.transform as lh_transform

from .builders import schedule_boilerplate


def remove_asserts(
    target_op: str | list[str] | None = None,
) -> ir.Module:
    """
    Removes asserts within the target Linalg ops and return handle
    to the updated ops.

    Args:
        target_op: Op(s) to consider. Defaults to all linalg ops.

    Returns:
        Schedule
    """
    if target_op is None:
        target_op = structured.MatchInterfaceEnum.LinalgOp

    with schedule_boilerplate() as (schedule, named_seq):
        ops = lh_transform.match_op(named_seq.bodyTarget, target_op)
        transform_ext.remove_asserts(ops)
        transform.yield_()
    return schedule

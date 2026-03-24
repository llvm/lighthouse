from mlir import ir
from mlir.dialects import transform

from .builders import schedule_boilerplate


def print_ir() -> ir.Module:
    """
    Print IR from the top-level module.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, _):
        transform.print_()
        transform.yield_()
    return schedule

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import math

from lighthouse.schedule.builders import schedule_boilerplate


def poly_approx() -> ir.Module:
    """
    Polynomial approximations for math functions.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        with ir.InsertionPoint(
            transform.ApplyPatternsOp(named_seq.bodyTarget).patterns
        ):
            math.ApplyF32ExpansionPatternsOp()
            math.ApplyPolynomialApproximationPatternsOp(enable_avx2=True)
        transform.yield_()
    return schedule

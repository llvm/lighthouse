from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import math, x86

from lighthouse.schedule.builders import schedule_boilerplate


def poly_approx(enable_x86: bool = False) -> ir.Module:
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
            math.ApplyPolynomialApproximationPatternsOp()
            if enable_x86:
                x86.ApplyMathRsqrtApproximationPatternsOp()
        transform.yield_()
    return schedule

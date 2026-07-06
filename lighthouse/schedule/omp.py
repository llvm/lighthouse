from mlir import ir
from mlir.dialects import transform

from lighthouse.pipeline.helper import apply_registered_pass
from lighthouse.schedule.builders import schedule_boilerplate


def parallelize() -> ir.Module:
    """
    Parallelizes execution using OpenMP.

    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        target = named_seq.bodyTarget
        target = apply_registered_pass(target, "scf-forall-to-parallel")
        apply_registered_pass(target, "convert-scf-to-openmp")
        transform.yield_()
    return schedule

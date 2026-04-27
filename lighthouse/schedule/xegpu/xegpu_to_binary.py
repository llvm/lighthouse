from mlir import ir
from mlir.dialects import transform

from ..builders import schedule_boilerplate
from lighthouse.pipeline.helper import apply_registered_pass


def xegpu_to_binary(xegpu_op_level: str = "workgroup") -> ir.Module:
    """
    Lower XeGPU dialect to binary using the default upstream pipeline.

    Args:
        xegpu_op_level: Initial XeGPU operation level.
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        target = named_seq.bodyTarget
        apply_registered_pass(
            target,
            "gpu-lower-to-xevm-pipeline",
            options={"xegpu-op-level": xegpu_op_level},
        )

        transform.yield_()
    return schedule

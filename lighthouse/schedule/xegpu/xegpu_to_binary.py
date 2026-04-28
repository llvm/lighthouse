from mlir import ir
from mlir.dialects import transform

from ..builders import schedule_boilerplate
from lighthouse.pipeline.helper import apply_registered_pass


def xegpu_to_binary(
    xegpu_op_level: str = "workgroup", large_register_file: bool = True
) -> ir.Module:
    """
    Lower XeGPU dialect to binary using the default upstream pipeline.

    Args:
        xegpu_op_level: Initial XeGPU operation level.
        large_register_file: Whether to enable large register file.
    Returns:
        Schedule
    """
    options = {
        "xegpu-op-level": xegpu_op_level,
    }
    if large_register_file:
        options["igc-cmd-options"] = "-ze-opt-large-register-file"
    with schedule_boilerplate() as (schedule, named_seq):
        target = named_seq.bodyTarget
        apply_registered_pass(
            target,
            "gpu-lower-to-xevm-pipeline",
            options=options,
        )

        transform.yield_()
    return schedule

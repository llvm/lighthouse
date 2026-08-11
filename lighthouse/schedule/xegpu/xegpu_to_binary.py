from mlir import ir
from mlir.dialects import transform

from ..builders import schedule_boilerplate
from lighthouse.pipeline.helper import apply_registered_pass


def xegpu_to_binary(
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
    enable_vector_to_xegpu: bool = True,
) -> ir.Module:
    """Build a transform schedule that lowers XeGPU IR to a device binary.

    Args:
        xegpu_op_level: Initial XeGPU operation level.
        large_register_file: Whether to enable large register file.
        enable_vector_to_xegpu: Whether to lower vector operations to XeGPU.

    Returns:
        Transform schedule module for ``gpu-lower-to-xevm-pipeline``.
    """
    options = {
        "xegpu-op-level": xegpu_op_level,
        "enable-vector-to-xegpu": "true" if enable_vector_to_xegpu else "false",
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

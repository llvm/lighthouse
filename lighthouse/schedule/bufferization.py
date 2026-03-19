from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import bufferization
from mlir.dialects.bufferization import LayoutMapOption

from .builders import schedule_boilerplate
import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import apply_registered_pass


def bufferize(deallocation_pipeline: bool = False) -> ir.Module:
    """
    Bufferize all ops.

    Args:
        deallocation_pipeline: Applies deallocation pipeline
    Returns:
        Schedule
    """
    with schedule_boilerplate() as (schedule, named_seq):
        target = named_seq.bodyTarget
        bufferization.bufferization_eliminate_empty_tensors(target)
        target = bufferization.bufferization_one_shot_bufferize(
            transform.any_op_t(),
            target,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
            bufferize_function_boundaries=True,
        )
        target = apply_registered_pass(target, "drop-equivalent-buffer-results")
        if deallocation_pipeline:
            target = apply_registered_pass(target, "buffer-deallocation-pipeline")
        target = apply_registered_pass(target, "convert-bufferization-to-memref")
        lh_transform.cleanup(target)

        transform.yield_()
    return schedule

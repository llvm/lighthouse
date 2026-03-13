from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import bufferization
from mlir.dialects.bufferization import LayoutMapOption

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import cleanup
from lighthouse.pipeline.helper import apply_registered_pass


def bufferize(deallocation_pipeline: bool = False) -> ir.Module:
    """
    Bufferize all ops.

    Args:
        deallocation_pipeline: Applies deallocation pipeline
    """
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
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
        cleanup(target)

        transform.yield_()
    return schedule

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import bufferization
from mlir.dialects.bufferization import LayoutMapOption

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import cleanup
from lighthouse.transform import apply_pass


def bufferize(deallocation_pipeline: bool = False) -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        bufferization.bufferization_eliminate_empty_tensors(named_seq.bodyTarget)
        bufferization.bufferization_one_shot_bufferize(
            named_seq.bodyTarget,
            function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
            bufferize_function_boundaries=True,
        )
        apply_pass(named_seq.bodyTarget, "drop-equivalent-buffer-results")
        if deallocation_pipeline:
            apply_pass(named_seq.bodyTarget, "buffer-deallocation-pipeline")
        apply_pass(named_seq.bodyTarget, "convert-bufferization-to-memref")
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

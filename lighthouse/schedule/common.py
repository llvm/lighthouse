from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from .builders import create_schedule
from .builders import create_named_sequence
from lighthouse.transform import loop_hoisting
from lighthouse.transform import cleanup


def schedule_loop_hoisting() -> ir.Module:
    schedule = create_schedule()
    named_seq = create_named_sequence(schedule, input_types=[transform.any_op_t()])

    with ir.InsertionPoint(named_seq.body):
        loop_hoisting(
            named_seq.bodyTarget, structured.MatchInterfaceEnum.LoopLikeInterface
        )
        cleanup(named_seq.bodyTarget)

        transform.yield_()
    return schedule

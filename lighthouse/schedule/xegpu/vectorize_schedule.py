from mlir import ir
from mlir.dialects import transform

from lighthouse.schedule import schedule_boilerplate
from . import lowering_common


def vectorize_schedule(
    payload_func_name: str | None = None,
) -> ir.Module:
    """Vectorizes the payload function."""

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        op_names = [
            "linalg.matmul",
            "linalg.generic",
            "vector.contract",
            "vector.transfer_read",
            "vector.transfer_write",
        ]
        func = lowering_common.get_payload_func(
            named_seq.bodyTarget, func_name=payload_func_name, op_name=op_names
        )
        payload_mod = transform.get_parent_op(
            anytype,
            func,
            op_name="builtin.module",
            deduplicate=True,
        )
        lowering_common.vectorize(payload_mod, payload_func=func)
        transform.yield_()

    return schedule

from mlir import ir
from mlir.dialects import transform

import lighthouse.transform as lh_transform
from lighthouse.pipeline.helper import apply_registered_pass
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
        func = lowering_common.vectorize(payload_mod, payload_func=func)
        func = apply_registered_pass(func, "remove-dead-values")
        lh_transform.cleanup(func)
        transform.yield_()

    return schedule

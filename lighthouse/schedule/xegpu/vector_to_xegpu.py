from mlir import ir
from mlir.dialects import transform

from lighthouse.schedule import schedule_boilerplate
from . import lowering_common
import lighthouse.transform as lh_transform


def vector_to_xegpu() -> ir.Module:
    """Converts vector ops in the outlined gpu.func to XeGPU ops."""

    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        op_names = [
            "linalg.matmul",
            "linalg.generic",
            "vector.contract",
            "vector.transfer_read",
            "vector.transfer_write",
        ]
        matching_children = lh_transform.match_op(named_seq.bodyTarget, op_names)
        payload_mod = transform.get_parent_op(
            anytype,
            matching_children,
            op_name="builtin.module",
            deduplicate=True,
        )
        payload_mod = lowering_common.convert_vector_to_xegpu(payload_mod)
        lh_transform.cleanup(payload_mod)
        transform.yield_()

    return schedule

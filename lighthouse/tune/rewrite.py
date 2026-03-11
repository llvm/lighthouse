from typing import Mapping

from mlir import ir
from mlir.dialects.transform import tune as transform_tune


def set_selected(op: ir.Operation, env: Mapping[ir.Value | ir.Operation, ir.Attribute]):
    """Walk op's IR and set attrs on transform.tune.* ops per the env mapping."""

    def set(op: ir.Operation) -> ir.WalkResult:
        op = op.opview
        match op:
            case transform_tune.KnobOp():
                op.attributes["selected"] = env[op.result]
            case transform_tune.AlternativesOp():
                op.attributes["selected_region"] = env[op]
        return ir.WalkResult.ADVANCE

    op.walk(set, ir.WalkOrder.PRE_ORDER)
    return op

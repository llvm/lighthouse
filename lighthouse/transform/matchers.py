from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def match_op(
    target,
    target_op: str | list[str] | structured.MatchInterfaceEnum,
) -> ir.Value:
    """
    Match ops by name or interface.

    Args:
        target: Handle to target
        target_op: Ops to be matched
    Returns:
        Handle to matched ops
    """
    if isinstance(target_op, structured.MatchInterfaceEnum):
        return structured.MatchOp(
            transform.any_op_t(),
            target,
            interface=target_op,
        )
    if isinstance(target_op, str):
        target_op = [target_op]
    return structured.MatchOp.match_op_names(target, target_op)

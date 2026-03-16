from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import loop
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


def cleanup(target):
    """
    Apply canonicalization patterns.

    Args:
        target: Handle to target
    """
    transform.apply_cse(target)
    with ir.InsertionPoint(transform.ApplyPatternsOp(target).patterns):
        transform.apply_patterns_canonicalization()


def loop_hoisting(target):
    """
    Apply loop hoisting.

    Args:
        target: Handle to target
    """
    transform.apply_licm(target)
    loop.loop_hoist_loop_invariant_subsets(target)

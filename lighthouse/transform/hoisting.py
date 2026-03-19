from mlir.dialects import transform
from mlir.dialects.transform import loop


def loop_hoisting(target):
    """
    Apply loop hoisting.

    Args:
        target: Handle to target
    """
    transform.apply_licm(target)
    loop.loop_hoist_loop_invariant_subsets(target)

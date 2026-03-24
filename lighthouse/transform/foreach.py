from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TypeVar

from mlir import ir
from mlir.dialects import transform


_Target = TypeVar("_Target", bound=ir.Operation | ir.Value | ir.OpView)


@contextmanager
def foreach(
    targets: _Target | Sequence[_Target],
    result_types: Sequence[ir.Type] | None = None,
    **foreach_kwarg,
) -> Iterator[tuple[ir.Operation, ir.BlockArgumentList]]:
    """
    Context decorator for creating foreach transform.

    Apply transforms nested under the foreach loop exactly once
    per element of the payload associated to the targets handle.

    The decorator creates a new foreach operation and places
    the insertion point in the loop's body.

    Args:
        target: Handle to targets or sequence of handles
        result_types: Result types (default: no returns)
        foreach_kwarg: Extra arguments for the foreach op
    Returns:
        Foreach operation and its body targets
    """
    if not isinstance(targets, Sequence):
        targets = [targets]
    if result_types is None:
        result_types = []

    foreach_op = transform.ForeachOp(result_types, targets, **foreach_kwarg)
    with ir.InsertionPoint(foreach_op.body):
        yield foreach_op, foreach_op.bodyTargets

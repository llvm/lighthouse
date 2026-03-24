from collections.abc import Sequence
from typing import TypeVar

from mlir import ir
from mlir.dialects import transform


_Target = TypeVar("_Target", bound=ir.Operation | ir.Value | ir.OpView)


class foreach(transform.ForeachOp):
    """
    Context manager wrapper for foreach transform op.

    Apply transforms nested under the foreach loop exactly once
    per element of the payload associated to the targets handle.

    The wrapper creates a new foreach operation.
    On entry, the insertion point is placed in the loop's body
    and the block arguments are returned.
    On exit, the insertion point is restored.

    Nested multiple entry is not supported.

    Typical usage:

        with foreach(ops_handle) as op:
            transform.rewrite(op)
            ...
            transform.yield_()

    With results:

        with (
            foreach_op := lh_transform.foreach(
                [linalg_ops, vec_ops], result_types=[type]
            )
        ) as (linalg_op, vec_op):
            ...
            transform.yield_([val])
        res = foreach_op.results[0]

    Args:
        targets: Handle to targets or sequence of handles
        result_types: Result types (default: no returns)
        with_zip_shortest: limit iterations to the shortest target
        kwargs: Additional arguments for the foreach operation
    """

    def __init__(
        self,
        targets: _Target | Sequence[_Target],
        result_types: Sequence[ir.Type] | None = None,
        *,
        with_zip_shortest: bool = False,
        **kwargs,
    ):
        if not isinstance(targets, Sequence):
            targets = [targets]
        if result_types is None:
            result_types = []

        super().__init__(
            results=result_types,
            targets=targets,
            with_zip_shortest=with_zip_shortest,
            **kwargs,
        )
        self.insertion_point: ir.InsertionPoint | None = None

    def __enter__(self) -> Sequence[ir.BlockArgument]:
        if self.insertion_point is not None:
            raise Exception("Nested re-entry is not supported")
        # Set insertion point in the loop's body
        self.insertion_point = ir.InsertionPoint(self.body)
        self.insertion_point.__enter__()

        return self.bodyTargets[0] if len(self.bodyTargets) == 1 else self.bodyTargets

    def __exit__(self, *args):
        # Restore insertion point
        self.insertion_point.__exit__(*args)
        self.insertion_point = None

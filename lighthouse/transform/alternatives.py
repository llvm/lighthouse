from collections.abc import Sequence

from mlir import ir
from mlir.dialects import transform


class alternatives(transform.AlternativesOp):
    """
    Context manager wrapper for the core transform.alternatives op.

    Attempts each region in order. If a transform in a region produces a
    silenceable failure, the changes are rolled back and the next region is
    attempted. This provides a "try/fallback" construct.

    The `scope` handle must point to an op that is isolated from above (e.g. a
    func.func), as failed regions are rolled back by cloning the scope. Each
    region receives a block argument mapped to the scope and must yield the same
    number and types of results as declared in `result_types`.

    Typical usage:

        alt = lh_transform.alternatives(func, num_alternatives=2,
                                        result_types=[func.type])
        with alt.region(0) as scope:
            ...
            transform.yield_([scope])
        with alt.region(1) as scope:
            transform.yield_([scope])
        result = alt.results[0]

    Args:
        scope: Handle to the isolated-from-above scope op.
        num_alternatives: Number of alternative regions to try.
        result_types: Result types yielded by each region (default: no returns).
        kwargs: Additional arguments for the alternatives operation.
    """

    def __init__(
        self,
        scope: ir.Value,
        num_alternatives: int,
        result_types: Sequence[ir.Type] | None = None,
        **kwargs,
    ):
        if result_types is None:
            result_types = []

        super().__init__(
            results_=result_types,
            num_alternatives=num_alternatives,
            scope=scope,
            **kwargs,
        )
        # The core alternatives binding does not create region blocks; add an
        # entry block taking the scope handle to each region.
        for region in self.regions:
            region.blocks.append(scope.type)

    def region(self, index: int) -> "_AlternativesRegion":
        return _AlternativesRegion(self.regions[index])


class _AlternativesRegion:
    """Context manager placing the insertion point inside one alternative region."""

    def __init__(self, region: ir.Region):
        self._block = region.blocks[0]
        self._insertion_point: ir.InsertionPoint | None = None

    def __enter__(self) -> ir.BlockArgument:
        self._insertion_point = ir.InsertionPoint(self._block)
        self._insertion_point.__enter__()
        return self._block.arguments[0]

    def __exit__(self, *args):
        self._insertion_point.__exit__(*args)
        self._insertion_point = None

from mlir import ir
from mlir.dialects import arith, ext, scf, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _collapse(loop, max_rank, rewriter):
    rank = len(loop.lowerBound)
    if rank <= max_rank:
        return loop.operation
    count = rank - max_rank + 1
    with ir.InsertionPoint(loop), loop.location:
        zero = arith.ConstantOp(ir.IndexType.get(), 0).result
        one = arith.ConstantOp(ir.IndexType.get(), 1).result
        trips = []
        total = one
        for lower, upper, step in zip(
            list(loop.lowerBound)[:count],
            list(loop.upperBound)[:count],
            list(loop.step)[:count],
        ):
            extent = arith.SubIOp(upper, lower).result
            extent = arith.MaxSIOp(extent, zero).result
            trip = arith.CeilDivSIOp(extent, step).result
            trips.append(trip)
            total = arith.MulIOp(total, trip).result
        collapsed = scf.ParallelOp(
            [],
            [zero, *list(loop.lowerBound)[count:]],
            [total, *list(loop.upperBound)[count:]],
            [one, *list(loop.step)[count:]],
            [],
        )
        for name in loop.attributes:
            if name != "operandSegmentSizes":
                collapsed.attributes[name] = loop.attributes[name]
    body = collapsed.region.blocks.append(*([ir.IndexType.get()] * max_rank))
    with ir.InsertionPoint(body), loop.location:
        linear = body.arguments[0]
        indices = {}
        for dim in reversed(range(count)):
            coordinate = arith.RemUIOp(linear, trips[dim]).result if dim else linear
            offset = arith.MulIOp(coordinate, loop.step[dim]).result
            indices[dim] = arith.AddIOp(offset, loop.lowerBound[dim]).result
            if dim:
                linear = arith.DivUIOp(linear, trips[dim]).result
        terminator = scf.ReduceOp([], 0)
    old_body = loop.region.blocks[0]
    replacements = [indices[dim] for dim in range(count)] + list(body.arguments[1:])
    for old, new in zip(old_body.arguments, replacements):
        old.replace_all_uses_with(new)
    for op in list(old_body.operations)[:-1]:
        op.operation.move_before(terminator)
    rewriter.erase_op(loop)
    return collapsed.operation


class CollapseParallelOp(TransformExtensionDialect.Operation, name="collapse_parallel"):
    """Collapse leading scf.parallel dimensions to at most `max_rank`.

    High-dimensional workloads can produce scf.parallel ops with more than
    three parallel dimensions. The default GPU loop mapping maps only the
    first three dimensions of an outermost parallel op to GPU thread blocks
    (workgroups) along x/y/z; conversion lowers the remaining dimensions to
    sequential scf.for loops inside each workgroup. Collapsing to max_rank=3
    before GPU mapping lets the entire parallel iteration space be distributed
    across workgroups rather than partially serialized.

    Uses row-major linearization within the collapsed dimension, reconstructing
    original indices, including nonzero bounds and strides.
    Loops already within the rank limit pass through unchanged. Targets must
    have signed bounds, no reductions and no existing mapping. Consumes loop
    handles and invalidates nested handles; returns the resulting loops.
    """

    target: ext.Operand[transform.AnyOpType]
    max_rank: ir.IntegerAttr = ext.attribute()
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state):
            max_rank = op.max_rank.value
            if max_rank < 1:
                op.location.emit_error("collapse_parallel: max_rank must be positive")
                return DiagnosedSilenceableFailure.DefiniteFailure
            targets = list(state.get_payload_ops(op.target))
            for target in targets:
                if (
                    target.name != "scf.parallel"
                    or len(target.results)
                    or "mapping" in target.attributes
                    or "unsignedCmp" in target.attributes
                ):
                    op.location.emit_error(
                        "collapse_parallel: expected unmapped scf.parallel "
                        "without reductions or unsigned bounds"
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure
            results.set_ops(
                op.result,
                [_collapse(target.opview, max_rank, rewriter) for target in targets],
            )
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def collapse_parallel(target: ir.Value, *, max_rank: int) -> ir.Value:
    """Collapse leading parallel dimensions, retaining at most `max_rank`."""
    return CollapseParallelOp(
        target=target,
        max_rank=ir.IntegerAttr.get(ir.IntegerType.get_signless(64), max_rank),
    ).result

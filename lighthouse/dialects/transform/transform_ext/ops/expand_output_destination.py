from mlir import ir
from mlir.dialects import ext, transform, memref, bufferization
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ExpandOutputDestinationOp(
    TransformExtensionDialect.Operation, name="expand_output_destination"
):
    """Remove a reshape from a result's write path so it bufferizes in place.

    Rewrites ``materialize_in_destination(collapse_shape(X), D)`` into
    ``materialize_in_destination(X, memref.expand_shape(D))``.

    WG-tiled attention writes its result in the GQA-expanded N-D layout, but the
    output memref is the collapsed (N-1)-D layout, so a ``tensor.collapse_shape``
    sits between the loop result and the destination. That reshape makes the loop
    result and the destination differ in shape, so empty-tensor elimination cannot
    fold the loop's output into the destination and bufferization materializes a
    separate buffer and copies it out. Reshaping the destination up to the loop's
    layout removes the reshape from the write path, so elimination fires and the
    loop writes the output in place.

    Only rewrites ``materialize_in_destination`` ops whose destination is a memref
    and whose source is a single ``tensor.collapse_shape``; a no-op otherwise.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, _rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = list(state.get_payload_ops(op.target))
            for func in targets:
                _rewrite(func)
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def _reassociation(collapse: ir.OpView) -> list[list[int]]:
    return [
        [ir.IntegerAttr(idx).value for idx in ir.ArrayAttr(group)]
        for group in ir.ArrayAttr(collapse.attributes["reassociation"])
    ]


def _rewrite(func: ir.OpView) -> None:
    materializes = [
        o
        for o in func.regions[0].blocks[0].operations
        if o.operation.name == "bufferization.materialize_in_destination"
    ]
    for mat in materializes:
        source, dest = mat.operands[0], mat.operands[1]
        if not isinstance(dest.type, ir.MemRefType):
            continue
        collapse = source.owner
        if collapse.operation.name != "tensor.collapse_shape":
            continue
        expanded = collapse.operands[0]
        expanded_type = ir.RankedTensorType(expanded.type)
        with ir.InsertionPoint(mat):
            dest_type = ir.MemRefType.get(
                expanded_type.shape, expanded_type.element_type
            )
            dest_expanded = memref.expand_shape(
                dest_type,
                dest,
                _reassociation(collapse),
                [],
                static_output_shape=list(expanded_type.shape),
            )
            bufferization.materialize_in_destination(
                None, expanded, dest_expanded, restrict=True, writable=True
            )
        mat.operation.erase()


def expand_output_destination(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Rewrite result ``materialize_in_destination`` to write in place."""
    return ExpandOutputDestinationOp(target=target).result

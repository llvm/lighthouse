from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ExtractHandleOp(TransformExtensionDialect.Operation, name="extract_handle"):
    """
    Returns the handle at the specified index in `target`.

    Args:
        target: Handle(s) to target op(s)
        index: Index of the handle to extract. Supports Python-style indexing.
    Returns:
        The handle at the specified index in `target`.
    """

    target: ext.Operand[transform.AnyOpType]
    index: ext.Operand[transform.AnyParamType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.result(infer_type=True)

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ExtractHandleOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            index_attr = state.get_params(op.index)
            if len(index_attr) == 1 and isinstance(index_attr[0], ir.IntegerAttr):
                index = index_attr[0].value
            else:
                return DiagnosedSilenceableFailure.SilenceableFailure

            n = len(target_ops)
            if index >= n or index < -n:
                raise IndexError(
                    f"extract_handle: Invalid index {index} for target of length {len(target_ops)}"
                )
            handle = target_ops[index]
            results.set_ops(op.ops, [handle])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ExtractHandleOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def extract_handle(
    target: ir.Value[transform.AnyOpType],
    index: int | ir.Value[transform.AnyParamType],
) -> ir.Value:
    """
    snake_case wrapper to create a ExtractHandleOp.

    Args:
        target: Handle(s) to target op(s)
        index: Index of the handle to extract. Supports Python-style indexing.
    Returns:
        The handle at the specified index in `target`.
    """
    if isinstance(index, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)
        index = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return ExtractHandleOp(target=target, index=index).result

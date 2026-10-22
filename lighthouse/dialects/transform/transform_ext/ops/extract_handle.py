from typing import TypeVar

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects._ods_common import get_op_result_or_value
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


HandleType = TypeVar("HandleType", bound=transform.AnyOpType | transform.AnyParamType)


class ExtractHandleOp(TransformExtensionDialect.Operation, name="extract_handle"):
    """
    Returns the handle at the specified index in `target`.

    Args:
        target: Handle associated with operations or parameters.
        index: Index of the handle to extract. Supports Python-style indexing.
        silenceable: If False (default), an out-of-range index raises a
            definite failure; if True, a silenceable failure.
    Returns:
        A handle of the same type as `target`, associated with its indexed element.
    """

    target: ext.Operand[HandleType]
    index: ext.Operand[transform.AnyParamType]
    silenceable: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)
    )
    ops: ext.Result[HandleType] = ext.result(default_factory=transform.AnyOpType.get)

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
            is_param = isinstance(op.target.type, transform.AnyParamType)
            targets = (
                state.get_params(op.target)
                if is_param
                else state.get_payload_ops(op.target)
            )
            index_attr = state.get_params(op.index)
            if len(index_attr) == 1 and isinstance(index_attr[0], ir.IntegerAttr):
                index = index_attr[0].value
            else:
                return DiagnosedSilenceableFailure.SilenceableFailure

            n = len(targets)
            if index >= n or index < -n:
                msg = (
                    f"extract_handle: invalid index {index} for target of length {n}. "
                )
                if op.silenceable.value:
                    # TODO attach `msg` once the silenceable-failure bindings
                    # support a message (emit_silenceable_error is unavailable).
                    return DiagnosedSilenceableFailure.SilenceableFailure
                op.location.emit_error(msg)
                return DiagnosedSilenceableFailure.DefiniteFailure
            handle = targets[index]
            if is_param:
                results.set_params(op.ops, [handle])
            else:
                results.set_ops(op.ops, [handle])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ExtractHandleOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def extract_handle(
    target: ir.Value[HandleType],
    index: int | ir.Value[transform.AnyParamType],
    silenceable: bool = False,
) -> ir.Value[HandleType]:
    """
    snake_case wrapper to create a ExtractHandleOp.

    Args:
        target: Handle associated with operations or parameters.
        index: Index of the handle to extract. Supports Python-style indexing.
        silenceable: If False (default), an out-of-range index raises a
            definite failure; if True, a silenceable failure.
    Returns:
        A handle of the same type as `target`, associated with its indexed element.
    """
    target = get_op_result_or_value(target)
    if isinstance(index, int):
        param_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), index)
        index = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return ExtractHandleOp(
        target=target,
        ops=target.type,
        index=index,
        silenceable=ir.IntegerAttr.get(
            ir.IntegerType.get_signless(1), int(silenceable)
        ),
    ).result

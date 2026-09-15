from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ReverseHandlesOp(TransformExtensionDialect.Operation, name="reverse_handles"):
    """
    Reverse the order of the payload ops associated with the target handle.

    Args:
        target: Handle to any number of ops.
    Returns:
        Handle to the same ops in reversed order.
    """

    target: ext.Operand[transform.AnyOpType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ReverseHandlesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = list(state.get_payload_ops(op.target))
            results.set_ops(op.ops, target_ops[::-1])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ReverseHandlesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def reverse_handles(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a ReverseHandlesOp.

    Args:
        target: Handle to any number of ops.
    Returns:
        Handle to the same ops in reversed order.
    """
    return ReverseHandlesOp(target=target).ops

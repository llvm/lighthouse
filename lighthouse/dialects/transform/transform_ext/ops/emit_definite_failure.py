from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class EmitDefiniteFailureOp(
    TransformExtensionDialect.Operation, name="emit_definite_failure"
):
    """
    Unconditionally emit a definite failure, aborting the transform interpreter.

    Args:
        target: Handle kept as an operand so the op has a well-defined position
            and is scheduled after the preceding (failing) matchers.
        message: Diagnostic message to emit.
    """

    target: ext.Operand[transform.AnyOpType]
    message: ir.StringAttr = ext.attribute(
        default_factory=lambda: ir.StringAttr.get("")
    )

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "EmitDefiniteFailureOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            op.location.emit_error(op.message.value or "all alternatives failed")
            return DiagnosedSilenceableFailure.DefiniteFailure

        @staticmethod
        def allow_repeated_handle_operands(_op: "EmitDefiniteFailureOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.only_reads_payload()
            )


def emit_definite_failure(
    target: ir.Value[transform.AnyOpType],
    message: str = "",
) -> None:
    """snake_case wrapper to create an EmitDefiniteFailureOp."""
    EmitDefiniteFailureOp(
        target=target,
        message=ir.StringAttr.get(message),
    )

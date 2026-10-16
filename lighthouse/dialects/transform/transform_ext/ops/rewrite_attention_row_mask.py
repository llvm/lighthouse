from mlir import ir
from mlir.dialects import ext, scf, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.attention_row_mask import (
    rewrite_attention_row_mask as _rewrite,
)


class RewriteAttentionRowMaskOp(
    TransformExtensionDialect.Operation, name="rewrite_attention_row_mask"
):
    """Replace all--inf row masks using guarded online attention's maximum.

    Run after dependent reduction fusion, before sinking score producers. A
    recognized ``all(scores == -inf)`` is replaced in PV by a comparison against
    the updated running maximum, and in post-loop users by the final maximum.
    PV's select must mask the guarded exponential to zero. Scores must be finite
    or -inf, and V finite, as required by the online attention fusion.

    Unmatched masks, unsupported uses and pre-loop users are preserved. With no
    matching mask this is a no-op, so the same schedule supports unmasked attention.
    Run DCE afterwards to remove dead mask reductions and operand slices.
    """

    reduction_loop: ext.Operand[transform.AnyOpType]
    rewritten_loop: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            loops = state.get_payload_ops(op.reduction_loop)
            if len(loops) != 1 or not isinstance(loops[0].opview, scf.ForOp):
                op.operation.location.emit_error(
                    "rewrite_attention_row_mask expects exactly one scf.for"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            _rewrite(loops[0].opview)
            results.set_ops(op.rewritten_loop, loops)
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


def rewrite_attention_row_mask(
    reduction_loop: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Rewrite supported row-mask uses without rebuilding the online loop."""
    return RewriteAttentionRowMaskOp(reduction_loop=reduction_loop).rewritten_loop

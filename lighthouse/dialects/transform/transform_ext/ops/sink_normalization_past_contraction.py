from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.contraction_normalization import (
    sink_normalization_past_contraction as _sink_norm,
)


class SinkNormalizationPastContractionOp(
    TransformExtensionDialect.Operation, name="sink_normalization_past_contraction"
):
    """
    Move a per-row scale from a contraction's operand to after the contraction.

    Rewrites ``contract(A / N, B)`` into ``contract(A, B) / N`` (and likewise for
    ``*``), which is legal exactly when ``N`` does not vary along the contraction's
    reduction axis, since then it factors out of the sum::

        sum_k (A[k] / N) * B[k]  ==  (sum_k A[k] * B[k]) / N

    Worth doing when the reduction is long: the scale moves from one per (row, k)
    element to one per (row, n) output element.

    A contraction qualifies when it has exactly one reduction dim, that dim is last,
    and its output map is the identity on the remaining parallel dims. The scaling
    producer must be an all-parallel two-input generic whose body is exactly one
    ``arith.divf``/``arith.mulf`` of two block arguments, with the contraction as its
    only user. Anything else is left alone.

    Args:
        target: Handle to root ops to work within (e.g. func.func).
    Returns:
        Handle to the same target roots.
    """

    target: ext.Operand[transform.AnyOpType]
    rewritten: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "SinkNormalizationPastContractionOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            for target in targets:
                _sink_norm(target, rewriter)
            results.set_ops(op.rewritten, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "SinkNormalizationPastContractionOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def sink_normalization_past_contraction(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create SinkNormalizationPastContractionOp."""
    op = SinkNormalizationPastContractionOp(target=target)
    return op.rewritten

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

    Rewrites ``contract(A / S, B)`` into ``contract(A, B) / S`` (and likewise for
    ``*``), which is legal when ``S`` does not vary along the contraction's
    reduction axis, since then it factors out of the sum:

        sum_k (A[k] / S) * B[k]  ==  (sum_k A[k] * B[k]) / S

    Worth doing when the reduction dim is very large compared to the other dims.

    The scale is expected inside the contraction's body, i.e. scale is already
    fused into the contraction's body.

    Before:
    ```
    %o = linalg.generic ins(%e, %l, %v) {                     (contraction over k)
           divf, mulf, addf }
    ```

    After:
    ```
    %o = linalg.generic ins(%e, %v) { mulf, addf }            (contraction over k)
    %n = linalg.generic ins(%o, %l) { arith.divf }            (all-parallel)
    ```

    Op checks that the sink applies and reports a silenceable error saying why if it does not. It
    requires:

      * Exactly one payload op for `contraction`;
      * `contraction` to be a `linalg.generic` with exactly one reduction dim, that
        dim innermost, and an identity output map;
      * its body to hold an ``arith.divf``/``arith.mulf`` on two input block
        arguments;
      * the scale operand not to reference the reduction dim -- the condition that
        lets it factor out;
      * the scale's element type to be convertible to the contraction's accumulator
        type (it is widened when the divide moves).

    Args:
        contraction: Handle to the contraction carrying the scale.
    Returns:
        rewritten_contraction: The contraction rebuilt without the scale operand.
        sunk_normalization: The new linalg.generic applying the scale after it.
    """

    contraction: ext.Operand[transform.AnyOpType]
    rewritten_contraction: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    sunk_normalization: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

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
            def reject(message: str) -> DiagnosedSilenceableFailure:
                op.operation.location.emit_error(
                    f"cannot sink the normalization past the contraction: {message}"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            targets = state.get_payload_ops(op.contraction)
            if len(targets) != 1:
                return reject(
                    f"expected exactly one payload op for 'contraction', got "
                    f"{len(targets)}"
                )

            result, error = _sink_norm(targets[0], rewriter)
            if error is not None:
                return reject(error)
            results.set_ops(op.rewritten_contraction, [result.contraction.operation])
            results.set_ops(op.sunk_normalization, [result.normalization.operation])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "SinkNormalizationPastContractionOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            # The in-body form rebuilds the contraction and erases the original, so
            # the handle is consumed; `rewritten_contraction` replaces it.
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def sink_normalization_past_contraction(
    contraction: ir.Value[transform.AnyOpType],
) -> tuple[ir.Value[transform.AnyOpType], ir.Value[transform.AnyOpType]]:
    """snake_case wrapper to create SinkNormalizationPastContractionOp."""
    op = SinkNormalizationPastContractionOp(contraction=contraction)
    return op.rewritten_contraction, op.sunk_normalization

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

    Worth doing when the reduction dim very large compared to the other dims.

    Before:
    ```
    %p = linalg.generic ins(%e, %l) { arith.divf }            (all-parallel)
    %o = linalg.generic ins(%p, %v) { mulf, addf }            (contraction over k)
    ```

    After:
    ```
    %o = linalg.generic ins(%e, %v) { mulf, addf }            (contraction over k)
    %n = linalg.generic ins(%o, %l) { arith.divf }            (all-parallel)
    ```

    Both payload ops are named explicitly. The op checks that the sink applies and
    reports a silenceable error saying why if it does not. It requires:

      * Exactly one payload op per handle;
      * `contraction` to be a structured linalg op (i.e. a `linalg.generic` or a named
        op such as ``linalg.matmul``/``linalg.batch_matmul``) with exactly one
        reduction dim, reduction dim is the innermost and, has an identity output map;
      * `normalization` to be an all-parallel two-input `linalg.generic` whose body
        is a single ``arith.divf``/``arith.mulf`` of two block arguments, with an
        identity output map with contraction as its only user;
      * The scale, seen from the contraction's iteration space, not to reference the
        reduction dim -- the condition that lets it factor out;
      * The scale's element type to be convertible to the contraction's accumulator
        type (it is widened when the divide moves).

    A named contraction additionally has to be able to read the numerator under the
    indexing map already in place, since its own verifier constrains those maps;
    generalize it to a `linalg.generic` first if not.

    `contraction` is rewritten in place, so a handle to it stays valid.

    Args:
        normalization: Handle to the scaling linalg.generic.
        contraction: Handle to the contraction consuming its result.
    Returns:
        Handle to the new linalg.generic applying the scale after the contraction.
    """

    normalization: ext.Operand[transform.AnyOpType]
    contraction: ext.Operand[transform.AnyOpType]
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

            payloads = []
            for name, handle in (
                ("normalization", op.normalization),
                ("contraction", op.contraction),
            ):
                handle_ops = state.get_payload_ops(handle)
                if len(handle_ops) != 1:
                    return reject(
                        f"expected exactly one payload op for '{name}', got "
                        f"{len(handle_ops)}"
                    )
                payloads.append(handle_ops[0])

            sunk, error = _sink_norm(payloads[0], payloads[1], rewriter)
            if error is not None:
                return reject(error)
            results.set_ops(op.sunk_normalization, [sunk.operation])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "SinkNormalizationPastContractionOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            # The normalization is erased, so its handle is consumed; the
            # contraction is rewritten in place, so its handle survives.
            operands = list(op.op_operands)
            return (
                transform.consumes_handle(operands[:1])
                + transform.only_reads_handle(operands[1:])
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def sink_normalization_past_contraction(
    normalization: ir.Value[transform.AnyOpType],
    contraction: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create SinkNormalizationPastContractionOp."""
    op = SinkNormalizationPastContractionOp(
        normalization=normalization, contraction=contraction
    )
    return op.sunk_normalization

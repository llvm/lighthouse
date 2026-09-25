from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.ops.fuse_elementwise_op_impl import (
    fuse_elementwise,
)


class FuseElementwiseOp(
    TransformExtensionDialect.Operation, name="fuse_elementwise_op"
):
    """Fuse an elementwise producer into one selected consumer.

    Both handles must identify a single ``linalg.generic``; the producer must
    have pure tensor semantics and only parallel loops. The producer must supply exactly one
    consumer input, not its output/init. Failure leaves the payload unchanged.

    The consumer handle is consumed and the fused op is returned. The producer
    and its other users are preserved; cleanup can remove a dead producer later.
    No unrelated operations are rewritten or canonicalized.

    Example:

    before
    ```
    E = exp(A)
    C = E + B
    ```

    after
    ```
    E = exp(A)
    C = exp(A) + B  // one fused linalg body
    ```
    """

    producer: ext.Operand[transform.AnyOpType]
    consumer: ext.Operand[transform.AnyOpType]
    fused: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            producers = list(state.get_payload_ops(op.producer))
            consumers = list(state.get_payload_ops(op.consumer))
            failure = DiagnosedSilenceableFailure.SilenceableFailure
            if len(producers) != 1 or len(consumers) != 1:
                return failure
            fused = fuse_elementwise(producers[0], consumers[0], rewriter)
            if fused is None:
                return failure
            results.set_ops(op.fused, [fused])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands[:1])
                + transform.consumes_handle(op.op_operands[1:])
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fuse_elementwise_op(
    producer: ir.Value[transform.AnyOpType],
    consumer: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """Fuse ``producer`` into ``consumer`` and return the fused consumer handle."""
    return FuseElementwiseOp(producer=producer, consumer=consumer).fused

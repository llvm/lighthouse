from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils.elementwise_chain_fusion import (
    fuse_same_rank_elementwise_chains as _fuse_chains,
)


class FuseSameRankElementwiseChainsOp(
    TransformExtensionDialect.Operation, name="fuse_same_rank_elementwise_chains"
):
    """
    Fuse chains of same-rank elementwise `linalg.generic` ops under `target`.

    An elementwise producer is merged into an elementwise consumer, inlining its
    body, when

      * both are all-parallel single-result generics whose output map is the
        identity,
      * the consumer reads the producer's result under an identity map -- same rank,
        no broadcast, no transpose,
      * the producer's body does not read its own `outs` argument, and
      * the consumer is the producer's only user, so fusing does not duplicate work.

    Repeats to a fixed point, so a chain of any length collapses into one op.

    Unlike ``linalg-fuse-elementwise-ops`` this never fuses across a reduction.
    Example:
        Input (softmax):
        %m = max_k %s        %d = %s - %m        %p = exp(%d)        %l = sum_k %p

        Output:
        %m = max_k %s        %pl = exp(%s - %m)        %l = sum_k %pl

        Only %d and %p are fused.

    Args:
        target: Handle to root ops to work within (e.g. func.func).
    Returns:
        Handle to the same target roots.
    """

    target: ext.Operand[transform.AnyOpType]
    fused: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FuseSameRankElementwiseChainsOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            for target in targets:
                _fuse_chains(target, rewriter)
            results.set_ops(op.fused, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "FuseSameRankElementwiseChainsOp",
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


def fuse_same_rank_elementwise_chains(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create FuseSameRankElementwiseChainsOp."""
    op = FuseSameRankElementwiseChainsOp(target=target)
    return op.fused

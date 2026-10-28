from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import subgraph_analysis as sa


class GetSubgraphOpsOp(TransformExtensionDialect.Operation, name="get_subgraph_ops"):
    """
    Get the ops of the subgraph anchored at `target` (see `tag_subgraphs`).

    Args:
        target: Handle to a single subgraph anchor op.
    Returns:
        Handles to the linalg producers of the anchor that belong to the
        subgraph, and to the subgraph ops that depend on the anchor, both in
        program order.
    """

    target: ext.Operand[transform.AnyOpType]
    producers: ext.Result[transform.AnyOpType[()]] = ext.infer_result()
    consumers: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetSubgraphOpsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            if (
                len(targets) != 1
                or sa.SUBGRAPH_ANCHOR_ATTR_NAME not in targets[0].attributes
            ):
                return DiagnosedSilenceableFailure.SilenceableFailure
            producers, consumers = sa.subgraph_members(targets[0].operation)
            results.set_ops(op.producers, producers)
            results.set_ops(op.consumers, consumers)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetSubgraphOpsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def get_subgraph_ops(
    target: ir.Value[transform.AnyOpType],
) -> tuple[ir.Value, ir.Value]:
    """
    snake_case wrapper to create a GetSubgraphOpsOp.

    Args:
        target: Handle to a single subgraph anchor op.
    Returns:
        Handles to the subgraph producers and consumers of the anchor.
    """
    op = GetSubgraphOpsOp(target=target)
    return op.producers, op.consumers

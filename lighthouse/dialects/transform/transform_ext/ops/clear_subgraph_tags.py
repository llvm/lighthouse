from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import subgraph_analysis as sa


class ClearSubgraphTagsOp(
    TransformExtensionDialect.Operation, name="clear_subgraph_tags"
):
    """
    Remove the `tag_subgraphs` attributes from the target op(s) and their
    nested ops.

    Args:
        target: Handle to op(s) to clear.
    Returns:
        Pass-through handle to the target op(s).
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ClearSubgraphTagsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            for target in targets:
                sa.clear_tags(target)
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ClearSubgraphTagsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def clear_subgraph_tags(target: ir.Value[transform.AnyOpType]) -> ir.Value:
    """
    snake_case wrapper to create a ClearSubgraphTagsOp.

    Args:
        target: Handle to op(s) to clear.
    Returns:
        Pass-through handle to the target op(s).
    """
    return ClearSubgraphTagsOp(target=target).result

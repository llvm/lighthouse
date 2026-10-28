from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import subgraph_analysis as sa


class TagSubgraphsOp(TransformExtensionDialect.Operation, name="tag_subgraphs"):
    """
    Partition the body of each target op into subgraphs and tag their ops.

    The matchers of `kinds` run in the given order and only claim untagged ops,
    so earlier kinds take precedence. Each subgraph op gets the
    `transform_ext.subgraph_kind` and `transform_ext.subgraph_id` attributes,
    and the op to tile additionally gets `transform_ext.subgraph_anchor`.
    Supported kinds:
      - "gemm": matmul with its single-use elementwise epilogue chain and the
        elementwise producers feeding it. Producers that also have users
        outside the subgraph are cloned so each subgraph recomputes them.
      - "elementwise": connected elementwise linalg ops and the reshapes,
        slices, and concats between them.

    Produces a silenceable failure if some linalg op in a target body is left
    untagged.

    Args:
        target: Handle to op(s) with a single-block region, e.g. func.func.
        kinds: Subgraph kinds, in matching order.
    Returns:
        Pass-through handle to the target op(s).
    """

    target: ext.Operand[transform.AnyOpType]
    kinds: ir.ArrayAttr = ext.attribute()
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "TagSubgraphsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            kinds = [ir.StringAttr(kind).value for kind in op.kinds]
            if any(kind not in sa.SUBGRAPH_MATCHERS for kind in kinds):
                op.location.emit_error(
                    f"tag_subgraphs: unknown kind in {kinds}, expected "
                    f"{sorted(sa.SUBGRAPH_MATCHERS)}"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure
            targets = state.get_payload_ops(op.target)
            untagged = []
            for target in targets:
                for region in target.regions:
                    for block in region.blocks:
                        untagged += sa.tag_subgraphs(block, kinds)
            results.set_ops(op.result, targets)
            if untagged:
                return DiagnosedSilenceableFailure.SilenceableFailure
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "TagSubgraphsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def tag_subgraphs(
    target: ir.Value[transform.AnyOpType],
    kinds: list[str],
) -> ir.Value:
    """
    snake_case wrapper to create a TagSubgraphsOp.

    Args:
        target: Handle to op(s) with a single-block region, e.g. func.func.
        kinds: Subgraph kinds, in matching order.
    Returns:
        Pass-through handle to the target op(s).
    """
    return TagSubgraphsOp(
        target=target,
        kinds=ir.ArrayAttr.get([ir.StringAttr.get(kind) for kind in kinds]),
    ).result

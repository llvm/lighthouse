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
      - "attention": closed QK-softmax-PV DAG and attention-local prologues.
        Match first to keep its reductions and elementwise ops together.
      - "reduction": a generic reduction with a closed elementwise
        normalization DAG, anchored at its final elementwise output. Match
        before GEMM to materialize normalization once rather than per GEMM tile.
        Shared input computations remain outside. Accumulator fills must
        already have been isolated by prepare_subgraphs when shared.
      - "gemm": matmul with its single-use elementwise epilogue chain and the
        elementwise producers feeding it. Shared producers selected for
        recomputation must already have been isolated by prepare_subgraphs.
        Token-only expand_shape and elementwise epilogue DAGs are fused forward.
        A join belongs to the last available branch: external inputs must
        already be defined at the root matmul, avoiding cyclic kernel ordering.
        Local elementwise setup may follow it if its inputs are available,
        e.g. bias casts/broadcasts and ReLU zero fills.
      - "elementwise": connected elementwise linalg ops and the reshapes,
        slices, and concats between them.

    Produces a silenceable failure if some linalg op in a target body is left
    untagged or shared producers require isolation. Run prepare_subgraphs with
    the same ordered kinds first when normalization/isolation is needed.
    Only scheduling attributes are changed: operation identity, order, operands,
    result types and scalar bodies are preserved, as are existing handles.

    Args:
        target: Handle to op(s) with a single-block region, e.g. func.func.
        kinds: Subgraph kinds, in matching order.
    """

    target: ext.Operand[transform.AnyOpType]
    kinds: ir.ArrayAttr = ext.attribute()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "TagSubgraphsOp",
            _rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
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
            if any(
                len(target.regions) != 1 or len(target.regions[0].blocks) != 1
                for target in targets
            ):
                op.location.emit_error(
                    "tag_subgraphs: expected targets with one single-block region"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            untagged = []
            for target in targets:
                for region in target.regions:
                    for block in region.blocks:
                        untagged += sa.tag_subgraphs(block, kinds)
            if untagged:
                op.location.emit_error(
                    "tag_subgraphs: unpartitioned ops or shared producers; "
                    "run prepare_subgraphs before tagging"
                )
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
                + transform.modifies_payload()
            )


def tag_subgraphs(
    target: ir.Value[transform.AnyOpType],
    kinds: list[str],
) -> None:
    """
    snake_case wrapper to create a TagSubgraphsOp.

    Args:
        target: Handle to op(s) with a single-block region, e.g. func.func.
        kinds: Subgraph kinds, in matching order.
    """
    TagSubgraphsOp(
        target=target,
        kinds=ir.ArrayAttr.get([ir.StringAttr.get(kind) for kind in kinds]),
    )

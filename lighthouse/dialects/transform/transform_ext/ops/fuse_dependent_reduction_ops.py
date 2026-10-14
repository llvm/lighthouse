import sys

from mlir import ir
from mlir.dialects import ext, linalg, scf, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import (
    dependent_reduction_legality as legality,
)
from lighthouse.dialects.transform.transform_ext.utils.dependent_reduction_fusion import (
    fuse_dependent_reduction_ops as apply_fusion,
    plan_correction_factor,
)
from lighthouse.dialects.transform.transform_ext.utils.elementwise_fusion import (
    build_fused_elementwise,
)
from lighthouse.dialects.transform.transform_ext.utils.ir_rewrite import depends_on_op
from lighthouse.utils.mlir import defining_op, linalg_inputs


def _prepare_consumer(e, r2):
    """Inline the elementwise path from E into a temporary R2 for analysis.

    Return the temporary candidate (or None without intermediates) and the
    intermediate ops ordered from E to R2. The loop is built from the original
    ops so intermediates stay outside R2's body.
    """
    candidate = None
    consumer = r2
    chain = []
    try:
        while e.results[0] not in linalg_inputs(consumer):
            edges = [
                value for value in linalg_inputs(consumer) if depends_on_op(value, e)
            ]
            if len(edges) != 1:
                raise legality.FusionRejected(
                    "expected one elementwise path from E to R2"
                )
            producer = defining_op(edges[0])
            if producer is None or producer.block != r2.operation.block:
                raise legality.FusionRejected(
                    "intermediate elementwise operations must share R2's block"
                )
            fused = build_fused_elementwise(producer, consumer)
            if fused is None:
                raise legality.FusionRejected(
                    "cannot inline the intermediate producer into R2"
                )
            chain.insert(0, producer.opview)
            if candidate is not None:
                candidate.operation.erase()
            candidate = consumer = fused
        if candidate is not None:
            ir.InsertionPoint(r2).insert(candidate.operation)
            try:
                candidate.operation.verify()
            except ir.MLIRError as error:
                raise legality.FusionRejected(
                    f"inlining the elementwise chain produced invalid IR: {error}"
                ) from error
        result = candidate
        candidate = None
        return result, chain
    finally:
        if candidate is not None:
            candidate.operation.erase()


def _single(payload_ops, what: str):
    """The one payload op behind a handle, or None if the handle is not singular."""
    ops = list(payload_ops)
    if len(ops) != 1:
        print(
            f"fuse_dependent_reduction_ops: requires exactly one {what}, got "
            f"{len(ops)}",
            file=sys.stderr,
        )
        return None
    return ops[0].opview if isinstance(ops[0], ir.Operation) else ops[0]


class FuseDependentReductionOpsOp(
    TransformExtensionDialect.Operation, name="fuse_dependent_reduction_ops"
):
    """Fuse one ``R1 -> E -> R2`` chain into R1's tiled reduction loop.

    R1 is supplied as a tiled ``scf.for``, conventionally marked
    ``__reduction_loop__``; E and R2 are ``linalg.generic`` ops. E reads R1's
    result, and R2 reduces E's result, possibly through elementwise intermediates.
    The loop step supplies the tile size. Intermediates are inlined into a
    temporary consumer for analysis only. In the fused loop they are tiled as
    separate elementwise ops between E and R2, so R2 keeps its own body (e.g. a
    plain multiply-add contraction); their other users are preserved.

    Other users of E keep the original while a clone is fused. One call handles
    one R2. The returned loop retains any reduction marker for another call.

    Each handle must identify one op. Invalid handles or an illegal chain cause
    a silenceable failure.

    The payload is inspected at apply time to recognize ``exp(x - maximum(x))``.
    Scores must be finite or negative infinity and other R2 inputs (e.g. V) finite.
    Empty tiles contribute zero, and the correction is ``exp(m_old - m_new)``,
    or zero when the old maximum is negative infinity. Fully masked rows still
    require a guard on the final normalization. Other separable E/R1 patterns
    retain the generic correction. Independent zero masks are supported in R2,
    whether already fused or in a separate elementwise chain.

    Args:
        elementwise_op: Handle to the elementwise term ``E``.
        reduction_op: Handle to the consumer reduction ``R2``.
        tiled_reduction_loop: Handle to ``R1``'s already-tiled ``scf.for``.
    Returns:
        Handle to the fused loop.
    """

    elementwise_op: ext.Operand[transform.AnyOpType]
    reduction_op: ext.Operand[transform.AnyOpType]
    tiled_reduction_loop: ext.Operand[transform.AnyOpType]
    fused_loop: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "FuseDependentReductionOpsOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            r1_loop = _single(
                state.get_payload_ops(op.tiled_reduction_loop),
                "tiled reduction loop",
            )
            e = _single(state.get_payload_ops(op.elementwise_op), "elementwise op")
            r2 = _single(state.get_payload_ops(op.reduction_op), "reduction op")
            if r1_loop is None or e is None or r2 is None:
                return DiagnosedSilenceableFailure.SilenceableFailure

            if not isinstance(r1_loop, scf.ForOp):
                print(
                    "fuse_dependent_reduction_ops: expected the tiled reduction "
                    "loop to be an scf.for op",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            for name, candidate in (("elementwise", e), ("reduction", r2)):
                if not isinstance(candidate, linalg.GenericOp):
                    print(
                        f"fuse_dependent_reduction_ops: expected the {name} op to "
                        f"be a linalg.generic op",
                        file=sys.stderr,
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure

            if not legality.collect_inner_reduction_generics(r1_loop):
                print(
                    "fuse_dependent_reduction_ops: the reduction loop body does "
                    "not contain any reduction linalg.generic",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            result_to_inner = legality.map_loop_results_to_inner_reductions(r1_loop)
            candidate = None
            try:
                candidate, chain = _prepare_consumer(e, r2)
                consumer = candidate if candidate is not None else r2
                e_tiled_dim, tile_size = legality.check_legal_fusion_triple(
                    r1_loop, result_to_inner, e, consumer
                )
                correction_plan = plan_correction_factor(r1_loop, e, consumer)
                # The inlined candidate is only needed for analysis.
                if candidate is not None:
                    candidate.operation.erase()
                    candidate = None
                fused = apply_fusion(
                    rewriter,
                    r1_loop,
                    e,
                    r2,
                    e_tiled_dim,
                    tile_size,
                    correction_plan,
                    chain,
                )
            except legality.FusionRejected as rejected:
                print(
                    f"fuse_dependent_reduction_ops: could not fuse the elementwise "
                    f"op and the consumer reduction into the producer reduction "
                    f"loop -- {rejected}",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            finally:
                if candidate is not None:
                    candidate.operation.erase()

            results.set_ops(op.fused_loop, [fused])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FuseDependentReductionOpsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "FuseDependentReductionOpsOp"):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fuse_dependent_reduction_ops(
    elementwise_op: ir.Value[transform.AnyOpType],
    reduction_op: ir.Value[transform.AnyOpType],
    tiled_reduction_loop: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Fuse the chain using apply-time payload analysis.

    Recognized softmax uses stable correction, assumes finite auxiliary inputs
    (V) and finite or -inf scores, and requires a final fully-masked-row guard.
    """
    op = FuseDependentReductionOpsOp(
        elementwise_op=elementwise_op,
        reduction_op=reduction_op,
        tiled_reduction_loop=tiled_reduction_loop,
    )
    return op.fused_loop

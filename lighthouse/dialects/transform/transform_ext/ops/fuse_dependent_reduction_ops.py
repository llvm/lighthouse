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
    result, and R2 reduces E's result. The loop step supplies the tile size.

    Other users of E keep the original while a clone is fused. One call handles
    one R2.The returned loop retains any reduction marker for another call.

    Each handle must identify one op. Invalid handles or an illegal chain cause
    a silenceable failure.

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
            try:
                e_tiled_dim, tile_size = legality.check_legal_fusion_triple(
                    r1_loop, result_to_inner, e, r2
                )
                correction_plan = plan_correction_factor(r1_loop, e, r2)
                fused = apply_fusion(
                    rewriter, r1_loop, e, r2, e_tiled_dim, tile_size, correction_plan
                )
            except legality.FusionRejected as rejected:
                print(
                    f"fuse_dependent_reduction_ops: could not fuse the elementwise "
                    f"op and the consumer reduction into the producer reduction "
                    f"loop -- {rejected}",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

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
    """Create the fusion transform and return its new loop handle."""
    op = FuseDependentReductionOpsOp(
        elementwise_op=elementwise_op,
        reduction_op=reduction_op,
        tiled_reduction_loop=tiled_reduction_loop,
    )
    return op.fused_loop

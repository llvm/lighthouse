import sys

from mlir import ir
from mlir.dialects import ext, linalg, scf, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import (
    dependant_reduction_legality as legality,
)
from lighthouse.dialects.transform.transform_ext.utils.dependant_reduction_fusion import (
    fuse_dependant_reduction_ops as apply_fusion,
)


def _single(payload_ops, what: str):
    """The one payload op behind a handle, or None if the handle is not singular."""
    ops = list(payload_ops)
    if len(ops) != 1:
        print(
            f"fuse_dependant_reduction_ops: requires exactly one {what}, got "
            f"{len(ops)}",
            file=sys.stderr,
        )
        return None
    return ops[0].opview if isinstance(ops[0], ir.Operation) else ops[0]


class FuseDependantReductionOpsOp(
    TransformExtensionDialect.Operation, name="fuse_dependant_reduction_ops"
):
    """
    Fuses a dependency chain ``R1 -> E -> R2`` into a single online (one-pass)
    reduction loop, where ``R1`` is the producer reduction (`tiled_reduction_loop`),
    ``E`` is an elementwise op (`elementwise_op`) consuming ``R1``'s result as a
    broadcast input, and ``R2`` is the consumer reduction (`reduction_op`) reducing
    ``E``'s output over the same axis.

    ``E`` is kept as a *separate* elementwise op rather than folded into ``R2``'s
    body; this keeps the per-element term explicit and easier to vectorize later.
    Because ``E`` is exactly the per-element term that ``R2`` reduces, the online
    correction factor is derived directly from ``E``.

    The producer must already be tiled along the shared reduction dimension into
    an ``scf.for`` annotated with the ``__reduction_loop__`` unit attribute; the
    tile size is read from that loop's step. This op does not tile the producer
    itself -- use ``transform.structured.tile_using_for`` followed by
    ``transform.annotate`` for that.
    TODO: Remove the annotation requirement for the producer reduction loop.

    Whichever copy is fused computes each tile against the *running* ``R1``
    accumulator, which the online correction only accounts for on ``R2``'s
    accumulator, so its own full-extent result is stale in every tile but the last
    and must not be read off the loop. That result surfaces as a dead loop result,
    which the schedule drops with ``remove-dead-values`` after vectorization.

    So if ``E`` has users besides ``R2``, this op fuses a *clone* and leaves the
    original where it is: there it still reads ``R1``'s final result off the loop and
    recomputes the term correctly for those users. With ``R2`` as its only user,
    ``E`` itself is fused and nothing is left behind.

    One application fuses one consumer reduction. ``E`` may have several, though --
    any consumer reducing it over the same axis is a candidate ``R2`` in its own
    right -- so applying the op once per consumer folds an attention chain (one
    ``exp`` term feeding both a row sum and a `P @ V` contraction) into a single loop,
    each application adding its own online correction. The returned loop keeps the
    ``__reduction_loop__`` attribute, so it can be passed straight back in as
    `tiled_reduction_loop` with no re-annotation.

    For more details about the fusion algorithm refer:
    https://discourse.llvm.org/t/linalg-rfc-an-approach-for-tiling-and-fusing-dependent-reductions-in-linalg/91698

    Return modes:
        Each handle must point to exactly one payload op, otherwise this produces
        a silenceable failure -- as it does when the loop is not an ``scf.for``,
        when either op is not a ``linalg.generic``, or when the triple does not
        satisfy the fusion legality conditions. The rejected condition is reported
        on stderr.

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
            op: "FuseDependantReductionOpsOp",
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
                    "fuse_dependant_reduction_ops: expected the tiled reduction "
                    "loop to be an scf.for op",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            for name, candidate in (("elementwise", e), ("reduction", r2)):
                if not isinstance(candidate, linalg.GenericOp):
                    print(
                        f"fuse_dependant_reduction_ops: expected the {name} op to "
                        f"be a linalg.generic op",
                        file=sys.stderr,
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure

            if not legality.collect_inner_reduction_generics(r1_loop):
                print(
                    "fuse_dependant_reduction_ops: the reduction loop body does "
                    "not contain any reduction linalg.generic",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            result_to_inner = legality.map_loop_results_to_inner_reductions(r1_loop)
            try:
                e_tiled_dim, tile_size = legality.check_legal_fusion_triple(
                    r1_loop, result_to_inner, e, r2
                )
                fused = apply_fusion(rewriter, r1_loop, e, r2, e_tiled_dim, tile_size)
            except legality.FusionRejected as rejected:
                print(
                    f"fuse_dependant_reduction_ops: could not fuse the elementwise "
                    f"op and the consumer reduction into the producer reduction "
                    f"loop -- {rejected}",
                    file=sys.stderr,
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            results.set_ops(op.fused_loop, [fused])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "FuseDependantReductionOpsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "FuseDependantReductionOpsOp"):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fuse_dependant_reduction_ops(
    elementwise_op: ir.Value[transform.AnyOpType],
    reduction_op: ir.Value[transform.AnyOpType],
    tiled_reduction_loop: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """
    snake_case wrapper to create a FuseDependantReductionOpsOp.

    Args:
        elementwise_op: Handle to the elementwise term ``E``.
        reduction_op: Handle to the consumer reduction ``R2``.
        tiled_reduction_loop: Handle to ``R1``'s already-tiled ``scf.for``.
    Returns:
        Handle to the fused loop.
    """
    op = FuseDependantReductionOpsOp(
        elementwise_op=elementwise_op,
        reduction_op=reduction_op,
        tiled_reduction_loop=tiled_reduction_loop,
    )
    return op.fused_loop

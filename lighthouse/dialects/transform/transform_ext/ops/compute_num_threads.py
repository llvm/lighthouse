from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ComputeNumThreadsOp(
    TransformExtensionDialect.Operation, name="compute_num_threads"
):
    """
    Compute the workgroup thread count from wg and sg tile params.

    Returns `base * prod_i(wg_i // sg_i)` over the tile dimensions, treating an
    untiled (0) dim as 1 to avoid division by zero. Both tile params are read at
    apply time, so their length (the tile rank) may vary with the payload.

    Args:
        wg_tile: Param associated with the workgroup tile sizes (one i64/dim).
        sg_tile: Param associated with the subgroup tile sizes (one i64/dim).
        base: Multiplier applied to the subgroup count (e.g. the subgroup size).
    Return:
        Param holding the thread count as a single i64.
    """

    wg_tile: ext.Operand[transform.AnyParamType]
    sg_tile: ext.Operand[transform.AnyParamType]
    base: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 1)
    )
    num_threads: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ComputeNumThreadsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            wg = [ir.IntegerAttr(a).value for a in state.get_params(op.wg_tile)]
            sg = [ir.IntegerAttr(a).value for a in state.get_params(op.sg_tile)]
            if len(wg) != len(sg):
                return DiagnosedSilenceableFailure.SilenceableFailure

            count = op.base.value
            for w, s in zip(wg, sg):
                # Treat an untiled (0) dim as 1 to avoid division by zero.
                count *= (w or 1) // (s or 1)

            i64 = ir.IntegerType.get_signless(64)
            results.set_params(op.num_threads, [ir.IntegerAttr.get(i64, count)])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ComputeNumThreadsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def compute_num_threads(
    wg_tile: ir.Value[transform.AnyParamType],
    sg_tile: ir.Value[transform.AnyParamType],
    base: int | ir.IntegerAttr = 1,
) -> ir.Value[transform.AnyParamType]:
    """
    snake_case wrapper to create a ComputeNumThreadsOp.

    Args:
        wg_tile: Param holding the workgroup tile sizes (one i64 per dim).
        sg_tile: Param holding the subgroup tile sizes (one i64 per dim).
        base: Multiplier applied to the subgroup count (e.g. the subgroup size).
    Return:
        Param holding the thread count as a single i64.
    """
    if not isinstance(base, ir.IntegerAttr):
        base = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), base)
    return ComputeNumThreadsOp(wg_tile=wg_tile, sg_tile=sg_tile, base=base).num_threads

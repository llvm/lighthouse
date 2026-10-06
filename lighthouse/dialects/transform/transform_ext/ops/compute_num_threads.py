from collections.abc import Sequence

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
    untiled (0) dim as 1 to avoid division by zero.

    The wg and sg tiles are each passed as one or more param operands via the
    single `tiles` operand list; `num_wg_handles` marks how many leading
    operands make up the wg tile (the rest make up the sg tile). Each operand
    may itself carry several i64 attrs, so the tiles may be supplied either as
    one param-per-dim handle or as a single handle holding all dims. All params
    are read at apply time, so the tile rank may vary with the payload.

    Args:
        tiles: wg tile param operand(s) followed by sg tile param operand(s).
        num_wg_handles: Number of leading `tiles` operands forming the wg tile.
        base: Multiplier applied to the subgroup count (e.g. the subgroup size).
    Return:
        Param holding the thread count as a single i64.
    """

    tiles: Sequence[ext.Operand[transform.AnyParamType]]
    num_wg_handles: ir.IntegerAttr
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
            operands = list(op.tiles)
            num_wg = op.num_wg_handles.value
            wg = [
                ir.IntegerAttr(a).value
                for handle in operands[:num_wg]
                for a in state.get_params(handle)
            ]
            sg = [
                ir.IntegerAttr(a).value
                for handle in operands[num_wg:]
                for a in state.get_params(handle)
            ]
            if len(wg) != len(sg):
                return DiagnosedSilenceableFailure.SilenceableFailure

            count = op.base.value
            for w, s in zip(wg, sg):
                # Treat an untiled (0) dim as 1 to avoid division by zero.
                factor = (w or 1) // (s or 1)
                assert factor > 0
                count *= factor

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
    wg_tile: ir.Value[transform.AnyParamType] | Sequence[ir.Value],
    sg_tile: ir.Value[transform.AnyParamType] | Sequence[ir.Value],
    base: int | ir.IntegerAttr = 1,
) -> ir.Value[transform.AnyParamType]:
    """
    snake_case wrapper to create a ComputeNumThreadsOp.

    Args:
        wg_tile: Param(s) holding the workgroup tile sizes. Either a single
            handle with one i64 per dim, or a list of per-dim handles.
        sg_tile: Param(s) holding the subgroup tile sizes, in the same layout as
            `wg_tile`.
        base: Multiplier applied to the subgroup count (e.g. the subgroup size).
    Return:
        Param holding the thread count as a single i64.
    """
    i64 = ir.IntegerType.get_signless(64)
    wg_handles = list(wg_tile) if isinstance(wg_tile, (list, tuple)) else [wg_tile]
    sg_handles = list(sg_tile) if isinstance(sg_tile, (list, tuple)) else [sg_tile]
    if not isinstance(base, ir.IntegerAttr):
        base = ir.IntegerAttr.get(i64, base)
    return ComputeNumThreadsOp(
        tiles=wg_handles + sg_handles,
        num_wg_handles=ir.IntegerAttr.get(i64, len(wg_handles)),
        base=base,
    ).num_threads

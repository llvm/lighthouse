from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class InferXeGPUReductionParamsOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_reduction_params"
):
    """
    Infer XeGPU reduction tiling parameters for a reduction anchor op.

    Returns the workgroup, subgroup, and reduction tile sizes as three params.
    Each result is a param associated with one i64 per iteration dimension (in
    loop order), so it can be passed directly to the tiling routines without
    splitting. The tile lengths depend on the problem (e.g. `[64, 0]` or
    `[0, 0, 64, 64]`).

    NOTE: placeholder implementation; the tiles are hard-coded for now and the
    analysis deriving them from `target` will be added in a follow-up.

    Args:
        target: Handle to the reduction anchor op(s).
    Return:
        Params holding the wg, sg, and reduction tile sizes.
    """

    target: ext.Operand[transform.AnyOpType]
    wg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    sg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    reduction_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def _size_attrs(sizes: list[int]) -> list[ir.IntegerAttr]:
        i64 = ir.IntegerType.get_signless(64)
        return [ir.IntegerAttr.get(i64, size) for size in sizes]

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "InferXeGPUReductionParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            # TODO: derive the tiles from `target`; placeholders for now.
            size_attrs = InferXeGPUReductionParamsOp._size_attrs
            results.set_params(op.wg_tile, size_attrs([64, 0]))
            results.set_params(op.sg_tile, size_attrs([8, 0]))
            results.set_params(op.reduction_tile, size_attrs([0, 32]))
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "InferXeGPUReductionParamsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def infer_xegpu_reduction_params(
    target: ir.Value[transform.AnyOpType],
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    snake_case wrapper to create an InferXeGPUReductionParamsOp.

    Args:
        target: Handle to the reduction anchor op(s).
    Return:
        Tuple of params holding the wg, sg, and reduction tile sizes. Each is a
        param associated with one i64 per iteration dimension, ready to pass to
        the tiling routines.
    """
    op = InferXeGPUReductionParamsOp(target=target)
    return op.wg_tile, op.sg_tile, op.reduction_tile

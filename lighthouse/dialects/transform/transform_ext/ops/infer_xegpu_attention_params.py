from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class InferXeGPUAttentionParamsOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_attention_params"
):
    """
    Infer XeGPU attention tiling parameters for an attention anchor op.

    Returns the workgroup tile size, the subgroup tile size, and the reduction
    (K/V sequence) tile size. `wg_tile` and `sg_tile` are params associated with
    one i64 per iteration dimension (in loop order), so they can be passed
    directly to the tiling routines without splitting; `reduction_tile` is a
    scalar i64 param.

    NOTE: placeholder implementation; the tiles are hard-coded for now and the
    analysis deriving them from `target` will be added in a follow-up. Later
    schedules will additionally need sg_rows, q/v load tiles and prefetch
    parameters; those results will be added here when required.

    Args:
        target: Handle to the attention anchor op(s).
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
            op: "InferXeGPUAttentionParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            # TODO: derive the tiles from `target`; placeholders for now.
            i64 = ir.IntegerType.get_signless(64)
            size_attrs = InferXeGPUAttentionParamsOp._size_attrs
            results.set_params(op.wg_tile, size_attrs([1, 1, 128]))
            results.set_params(op.sg_tile, size_attrs([0, 0, 16]))
            # reduction_tile is a scalar; generalize to an array later if needed.
            results.set_params(op.reduction_tile, [ir.IntegerAttr.get(i64, 64)])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "InferXeGPUAttentionParamsOp",
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


def infer_xegpu_attention_params(
    target: ir.Value[transform.AnyOpType],
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    snake_case wrapper to create an InferXeGPUAttentionParamsOp.

    Args:
        target: Handle to the attention anchor op(s).
    Return:
        Tuple of params holding the wg and sg tile sizes (one i64 per iteration
        dimension, ready to pass to the tiling routines) and the scalar
        reduction tile size.
    """
    op = InferXeGPUAttentionParamsOp(target=target)
    return op.wg_tile, op.sg_tile, op.reduction_tile

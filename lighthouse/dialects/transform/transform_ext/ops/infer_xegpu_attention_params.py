from mlir import ir
from mlir.dialects import ext, transform, linalg, vector
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

    NOTE: This is just a placeholder implementation with hard-coded tile sizes.

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
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                # expecting a single anchor op
                return DiagnosedSilenceableFailure.SilenceableFailure
            target_op = target_ops[0]
            rank = ir.ShapedType(target_op.results[0].type).rank

            size_attrs = InferXeGPUAttentionParamsOp._size_attrs

            wg_rows, sg_rows, reduction_tile = 128, 16, 64

            if isinstance(target_op, linalg.GenericOp):
                # Assume non-tiled linalg.matmul
                # WG/SG row sizes and the reduction (K/V seq) tile are hard-coded.
                # Tile every leading parallel dim (incl. the GQA group) by 1, the
                # query-row dim by the WG row size, and leave d_head untiled.
                if rank == 4:  # plain MHA: (batch, head, query_row, d_head)
                    wg, sg = [1, 1, wg_rows], [0, 0, sg_rows]
                    red = [0, 0, 0, reduction_tile]
                elif rank == 5:  # GQA: (batch, kv_head, group, query_row, d_head)
                    wg, sg = [1, 1, 1, wg_rows], [0, 0, 0, sg_rows]
                    red = [0, 0, 0, 0, reduction_tile]
                else:
                    op.location.emit_error(
                        "infer_xegpu_attention_params: unsupported attention leaf "
                        f"rank {rank}; expected 4 (MHA) or 5 (GQA)"
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure
            elif isinstance(target_op, vector.ContractionOp) and rank == 2:
                # Assume tiled vector.contract, op's result shape is
                # (wg_rows, reduction_tile)
                wg_rows, _ = ir.ShapedType(target_op.results[0].type).shape
                wg = [wg_rows, 0]
                sg = [sg_rows, 0]
                red = [0, reduction_tile]
            else:
                op.location.emit_error(
                    "infer_xegpu_attention_params: unsupported attention leaf "
                    f"op {target_op.operation.name} with rank {rank}"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            results.set_params(op.wg_tile, size_attrs(wg))
            results.set_params(op.sg_tile, size_attrs(sg))
            results.set_params(op.reduction_tile, size_attrs(red))
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

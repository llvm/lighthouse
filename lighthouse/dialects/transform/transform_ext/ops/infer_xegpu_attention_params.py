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
    list of i64 params in loop order.

    NOTE: This is just a placeholder implementation with hard-coded tile sizes.

    Args:
        target: Handle to the attention anchor op(s).
        force_wg_tile: Optional workgroup tile sizes, used verbatim.
        force_sg_tile: Optional subgroup tile sizes, used verbatim.
        force_reduction_tile: Optional reduction tile sizes, used verbatim.
    Return:
        Params holding the wg, sg, and reduction tile sizes.
    """

    target: ext.Operand[transform.AnyOpType]
    force_wg_tile: ext.Operand[transform.AnyParamType] | None = None
    force_sg_tile: ext.Operand[transform.AnyParamType] | None = None
    force_reduction_tile: ext.Operand[transform.AnyParamType] | None = None
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
            # TODO infer tile sizes from problem shape
            wg_rows, sg_rows, reduction_tile = 128, 16, 64

            if isinstance(target_op, linalg.GenericOp):
                # Assume non-tiled linalg.matmul
                # WG/SG row sizes and the reduction (K/V seq) tile are hard-coded.
                # Tile every leading parallel dim (incl. the GQA group) by 1, the
                # query-row dim by the WG row size, and leave d_head untiled.
                if rank == 3:  # plain 3D attention: (head, query_row, d_head)
                    wg, sg = [1, wg_rows], [0, sg_rows]
                    red = [0, 0, reduction_tile]
                elif rank == 4:  # plain MHA: (batch, head, query_row, d_head)
                    wg_rows, sg_rows, reduction_tile = 128, 16, 64
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
                wg_rows, reduction_tile = ir.ShapedType(target_op.results[0].type).shape
                # Tile sizes are now for 2D case
                wg = [wg_rows]
                sg = [sg_rows]
                red = [0, reduction_tile]
            else:
                op.location.emit_error(
                    "infer_xegpu_attention_params: unsupported attention leaf "
                    f"op {target_op.operation.name} with rank {rank}"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            if op.force_wg_tile is not None:
                forced_attr = state.get_params(op.force_wg_tile)
                _wg = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]
                wg = _wg[-len(wg) :]  # drop leading dims if inferred is shorter
                print(f"Overriding {wg=}")
            if op.force_sg_tile is not None:
                forced_attr = state.get_params(op.force_sg_tile)
                _sg = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]
                sg = _sg[-len(sg) :]  # drop leading dims if inferred is shorter
                print(f"Overriding {sg=}")
            if op.force_reduction_tile is not None:
                forced_attr = state.get_params(op.force_reduction_tile)
                _red = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]
                red = _red[-len(red) :]  # drop leading dims if inferred is shorter
                print(f"Overriding {red=}")

            size_attrs = InferXeGPUAttentionParamsOp._size_attrs
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
    force_wg_tile: list[int] | None = None,
    force_sg_tile: list[int] | None = None,
    force_reduction_tile: list[int] | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    snake_case wrapper to create an InferXeGPUAttentionParamsOp.

    Args:
        target: Handle to the attention anchor op(s).
        force_wg_tile: Optional workgroup tile sizes, used verbatim.
        force_sg_tile: Optional subgroup tile sizes, used verbatim.
        force_reduction_tile: Optional reduction tile sizes, used verbatim.
    Return:
        Tuple of params holding the wg and sg tile sizes (one i64 per iteration
        dimension, ready to pass to the tiling routines) and the reduction
        tile sizes in loop order.
    """
    wg_param = None
    if force_wg_tile is not None:
        wg_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(InferXeGPUAttentionParamsOp._size_attrs(force_wg_tile)),
        )
    sg_param = None
    if force_sg_tile is not None:
        sg_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(InferXeGPUAttentionParamsOp._size_attrs(force_sg_tile)),
        )
    reduction_param = None
    if force_reduction_tile is not None:
        reduction_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(
                InferXeGPUAttentionParamsOp._size_attrs(force_reduction_tile)
            ),
        )
    op = InferXeGPUAttentionParamsOp(
        target=target,
        force_wg_tile=wg_param,
        force_sg_tile=sg_param,
        force_reduction_tile=reduction_param,
    )
    return op.wg_tile, op.sg_tile, op.reduction_tile

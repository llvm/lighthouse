from mlir import ir
from mlir.dialects import ext, transform, vector, linalg
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
    Generic trailing reductions tile the last parallel dimension by 64 and
    earlier parallel dimensions by one; the reduction tile is 32. Other
    reduction iterator arrangements require the existing explicit overrides.

    Args:
        target: Handle to the reduction anchor op(s).
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
            op: "InferXeGPUReductionParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                # expecting a single anchor op
                return DiagnosedSilenceableFailure.SilenceableFailure
            target_op = target_ops[0]

            sg_default = 8
            wg_rows = 64
            red_size = 32
            if isinstance(target_op, linalg.GenericOp):
                # TODO Infer meaningful values.
                wg_rows = 64
                red_size = 32
            elif isinstance(target_op, vector.MultiDimReductionOp):
                input_shape = ir.ShapedType(target_op.operands[0].type).shape
                if len(input_shape) != 2:
                    op.location.emit_error(
                        "infer_xegpu_reduction_params: expected 2D input for MultiDimReductionOp"
                    )
                    return DiagnosedSilenceableFailure.SilenceableFailure
                wg_rows, red_size = input_shape
            else:
                op.location.emit_error(
                    "infer_xegpu_reduction_params: unsupported reduction leaf "
                    f"op {target_op.operation.name} with rank {len(ir.ShapedType(target_op.operands[0].type).shape)}"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure

            wg = [wg_rows, 0]
            sg = [sg_default, 0]
            red = [0, red_size]
            if isinstance(target_op, linalg.GenericOp) and any(
                str(it) == "#linalg.iterator_type<reduction>"
                for it in target_op.iterator_types
            ):
                iterators = [str(it) for it in target_op.iterator_types]
                if iterators.count("#linalg.iterator_type<reduction>") != 1 or (
                    iterators[-1] != "#linalg.iterator_type<reduction>"
                ):
                    if op.force_wg_tile is None or op.force_reduction_tile is None:
                        return DiagnosedSilenceableFailure.SilenceableFailure
                else:
                    rank = len(iterators)
                    if rank < 2:
                        return DiagnosedSilenceableFailure.SilenceableFailure
                    wg = [1] * (rank - 2) + [wg_rows, 0]
                    sg = [1] * (rank - 2) + [sg_default, 0]
                    red = [0] * (rank - 1) + [red_size]

            if op.force_wg_tile is not None:
                forced_attr = state.get_params(op.force_wg_tile)
                wg = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]
            if op.force_sg_tile is not None:
                forced_attr = state.get_params(op.force_sg_tile)
                sg = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]
            if op.force_reduction_tile is not None:
                forced_attr = state.get_params(op.force_reduction_tile)
                red = [ir.IntegerAttr(a).value for a in ir.ArrayAttr(forced_attr[0])]

            size_attrs = InferXeGPUReductionParamsOp._size_attrs
            results.set_params(op.wg_tile, size_attrs(wg))
            results.set_params(op.sg_tile, size_attrs(sg))
            results.set_params(op.reduction_tile, size_attrs(red))
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
    force_wg_tile: list[int] | None = None,
    force_sg_tile: list[int] | None = None,
    force_reduction_tile: list[int] | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    snake_case wrapper to create an InferXeGPUReductionParamsOp.

    Args:
        target: Handle to the reduction anchor op(s).
        force_wg_tile: Optional workgroup tile sizes, used verbatim.
        force_sg_tile: Optional subgroup tile sizes, used verbatim.
        force_reduction_tile: Optional reduction tile sizes, used verbatim.
    Return:
        Tuple of params holding the wg, sg, and reduction tile sizes. Each is a
        param associated with one i64 per iteration dimension, ready to pass to
        the tiling routines.
    """
    wg_param = None
    if force_wg_tile is not None:
        wg_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(InferXeGPUReductionParamsOp._size_attrs(force_wg_tile)),
        )
    sg_param = None
    if force_sg_tile is not None:
        sg_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(InferXeGPUReductionParamsOp._size_attrs(force_sg_tile)),
        )
    reduction_param = None
    if force_reduction_tile is not None:
        reduction_param = transform.ParamConstantOp(
            transform.AnyParamType.get(),
            ir.ArrayAttr.get(
                InferXeGPUReductionParamsOp._size_attrs(force_reduction_tile)
            ),
        )
    op = InferXeGPUReductionParamsOp(
        target=target,
        force_wg_tile=wg_param,
        force_sg_tile=sg_param,
        force_reduction_tile=reduction_param,
    )
    return op.wg_tile, op.sg_tile, op.reduction_tile

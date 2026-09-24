from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from ..utils.xegpu_param_selection import select_xegpu_gemm_params
from ..utils.matmul_analysis import analyze_wg_k_tile_size


class InferXeGPUGemmParamsOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_gemm_params"
):
    """
    Infer XeGPU GEMM parameters for a matmul-like anchor op.

    Reads the global (M, N, K) shape and transpose flags from the first anchor
    op (vector.contract or xegpu.dpas), queries the XeGPU parameter selector,
    and returns all selected parameters as a single dictionary param.

    Args:
        target: Handle to the anchor op(s); only the first is processed.
        device: Optional target device name (unset selects the default).
    Return:
        Param holding a dictionary attribute of all selected parameters.
    """

    target: ext.Operand[transform.AnyOpType]
    device: ir.StringAttr = ext.attribute(default_factory=lambda: ir.StringAttr.get(""))
    params: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "InferXeGPUGemmParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) == 0:
                return DiagnosedSilenceableFailure.SilenceableFailure

            anchor_op = target_ops[0].opview
            device = op.device.value or None
            try:
                wg_tile, k_tile = analyze_wg_k_tile_size(anchor_op)
                params = select_xegpu_gemm_params(
                    anchor_op, wg_tile=wg_tile, k_tile=k_tile, device=device
                )
            except (KeyError, ValueError, StopIteration, NotImplementedError) as e:
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Failed to infer XeGPU GEMM params: {e}"
                )

            i64 = ir.IntegerType.get_signless(64)
            entries = {}
            for key, value in params.items():
                # bool is a subclass of int, so it must be checked first; emit
                # it as an i64 0/1 to avoid unsupported bool params downstream.
                if isinstance(value, bool):
                    entries[key] = ir.IntegerAttr.get(i64, int(value))
                elif isinstance(value, int):
                    entries[key] = ir.IntegerAttr.get(i64, value)
                elif isinstance(value, str):
                    entries[key] = ir.StringAttr.get(value)

            results.set_params(op.params, [ir.DictAttr.get(entries)])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "InferXeGPUGemmParamsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def infer_xegpu_gemm_params(
    target: ir.Value[transform.AnyOpType],
    device: str | ir.StringAttr | None = None,
) -> ir.Value[transform.AnyParamType]:
    """
    snake_case wrapper to create an InferXeGPUGemmParamsOp.

    Args:
        target: Handle to the anchor op(s); only the first is processed.
        device: Optional target device name.
    Return:
        Param holding a dictionary attribute of all selected parameters.
    """
    if device is None:
        device = ir.StringAttr.get("")
    elif not isinstance(device, ir.StringAttr):
        device = ir.StringAttr.get(device)
    return InferXeGPUGemmParamsOp(target=target, device=device).params

from collections.abc import Sequence

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class InferXeGPUPrefetchTileOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_prefetch_tile"
):
    """Select a 2D prefetch tile for a static data shape and XeGPU device.

    The shape may be supplied as one parameter containing two integers or as
    two scalar parameters. The optional minimum is a heuristic constraint on
    the number of subgroup threads, not a hardware requirement.
    """

    shape: Sequence[ext.Operand[transform.AnyParamType]]
    device: ir.StringAttr = ext.attribute(default_factory=lambda: ir.StringAttr.get(""))
    min_threads: ir.IntegerAttr = ext.attribute(
        default_factory=lambda: ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)
    )
    rows: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    cols: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "InferXeGPUPrefetchTileOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            from lighthouse.schedule.xegpu.matmul_costmodel import (
                generate_prefetch_tiles_for_shape,
            )
            from lighthouse.schedule.xegpu.xegpu_specs import XeGPUSpecs

            attrs = [a for handle in op.shape for a in state.get_params(handle)]
            if len(attrs) != 2 or not all(isinstance(a, ir.IntegerAttr) for a in attrs):
                op.location.emit_error("infer_xegpu_prefetch_tile: expected a 2D shape")
                return DiagnosedSilenceableFailure.DefiniteFailure
            shape = (ir.IntegerAttr(attrs[0]).value, ir.IntegerAttr(attrs[1]).value)
            if any(dim <= 0 for dim in shape) or op.min_threads.value < 0:
                op.location.emit_error(
                    "infer_xegpu_prefetch_tile: shape must be positive and "
                    "min_threads must be non-negative"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure

            try:
                specs = XeGPUSpecs.get(op.device.value or "B70")
                tiles = generate_prefetch_tiles_for_shape(
                    shape, specs, min_nb_threads=op.min_threads.value or None
                )
            except ValueError as e:
                op.location.emit_error(f"infer_xegpu_prefetch_tile: {e}")
                return DiagnosedSilenceableFailure.DefiniteFailure
            if not tiles:
                op.location.emit_error(
                    f"infer_xegpu_prefetch_tile: no valid tiles for shape {shape}"
                )
                return DiagnosedSilenceableFailure.DefiniteFailure

            i64 = ir.IntegerType.get_signless(64)
            results.set_params(op.rows, [ir.IntegerAttr.get(i64, tiles[0][0])])
            results.set_params(op.cols, [ir.IntegerAttr.get(i64, tiles[0][1])])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "InferXeGPUPrefetchTileOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.only_reads_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.only_reads_payload()
            )


def infer_xegpu_prefetch_tile(
    shape: ir.Value[transform.AnyParamType] | Sequence[ir.Value],
    device: str | None = None,
    min_threads: int = 0,
) -> tuple[ir.Value, ir.Value]:
    """Return scalar row and column parameters for a prefetch tile."""
    i64 = ir.IntegerType.get_signless(64)
    op = InferXeGPUPrefetchTileOp(
        shape=list(shape) if isinstance(shape, (list, tuple)) else [shape],
        device=ir.StringAttr.get(device or ""),
        min_threads=ir.IntegerAttr.get(i64, min_threads),
    )
    return op.rows, op.cols

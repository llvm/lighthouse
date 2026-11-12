from mlir import ir
from mlir.dialects import ext, transform, vector, xegpu
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.ops.filter_elementwise import (
    is_elementwise,
)
from lighthouse.utils.mlir import indexing_maps

_WG_TILE = (128, 256)
_SG_TILE = (32, 32)


def _largest_pow2_divisor(extent: int, bound: int) -> int:
    size = 1
    while size * 2 <= bound and extent % (size * 2) == 0:
        size *= 2
    return size


def elementwise_wg_tile(shape: list[int]) -> list[int]:
    """Workgroup tile for an elementwise iteration space.

    Tiles the two innermost non-unit dims by at most 128 rows and 256 columns,
    and outer dims by one. Zero marks an untiled dim. Tiles divide the shape
    exactly, avoiding partial tiles in the tensor-level epilogue fusion.
    """
    tile = [1] * len(shape)
    dimensions = [i for i, size in enumerate(shape) if size > 1][-2:]
    for dim, bound in zip(reversed(dimensions), reversed(_WG_TILE)):
        tile[dim] = _largest_pow2_divisor(shape[dim], bound)
    return [0 if t == extent else t for t, extent in zip(tile, shape)]


def elementwise_sg_tile(shape: list[int]) -> list[int]:
    """Distribute a vector tile over SGs along its two innermost non-unit dims."""
    tile = list(shape)
    dimensions = [i for i, size in enumerate(shape) if size > 1][-2:]
    for dim, bound in zip(reversed(dimensions), reversed(_SG_TILE)):
        tile[dim] = _largest_pow2_divisor(shape[dim], bound)
    return tile


class InferXeGPUElementwiseParamsOp(
    TransformExtensionDialect.Operation, name="infer_xegpu_elementwise_params"
):
    """
    Infer elementwise workgroup and subgroup tile sizes.

    For a tensor linalg elementwise anchor, wg_tile is derived from its single
    result's positive static shape (see `elementwise_wg_tile`) and mapped into
    iteration order. The result indexing map must be a permutation. Zero marks
    an untiled dimension; sg_tile is left empty at this stage.

    For a vector.transfer_write, xegpu.load_nd or xegpu.store_nd anchor, wg_tile
    is the full vector shape and sg_tile distributes its two innermost non-unit
    dimensions over subgroups using tiles of at most 32 by 32 elements.

    Args:
        target: Handle to a single tensor elementwise or vector memory anchor.
    Return:
        wg_tile: One i64 per tile dimension.
        sg_tile: One i64 per tile dimension for vector anchors, empty for linalg.
    """

    target: ext.Operand[transform.AnyOpType]
    wg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()
    sg_tile: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "InferXeGPUElementwiseParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            if len(targets) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure
            target = targets[0]
            if isinstance(
                target, (vector.TransferWriteOp, xegpu.LoadNdOp, xegpu.StoreNdOp)
            ):
                value = (
                    target.results[0]
                    if isinstance(target, xegpu.LoadNdOp)
                    else target.operands[0]
                )
                ty = value.type
                if not isinstance(ty, ir.VectorType) or ty.scalable or not ty.rank:
                    return DiagnosedSilenceableFailure.SilenceableFailure
                i64 = ir.IntegerType.get_signless(64)
                tile = [ir.IntegerAttr.get(i64, size) for size in ty.shape]
                results.set_params(op.wg_tile, tile)
                results.set_params(
                    op.sg_tile,
                    [
                        ir.IntegerAttr.get(i64, size)
                        for size in elementwise_sg_tile(list(ty.shape))
                    ],
                )
                return DiagnosedSilenceableFailure.Success
            # target is a linalg elementwise op
            if len(target.results) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure
            ty = targets[0].results[0].type
            if (
                not isinstance(ty, ir.RankedTensorType)
                or not ty.rank
                or any(size <= 0 for size in ty.shape)
                or ty.encoding is not None
            ):
                op.location.emit_error(
                    "infer_xegpu_elementwise_params: expected a positive-rank, "
                    "positive static, unencoded tensor result"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            maps = indexing_maps(targets[0])
            if (
                not maps
                or not is_elementwise(targets[0])
                or not maps[-1].is_permutation
            ):
                op.location.emit_error(
                    "infer_xegpu_elementwise_params: expected an elementwise "
                    "linalg op with a permutation result map"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            tile = [0] * maps[-1].n_dims
            for dim, size in zip(maps[-1].results, elementwise_wg_tile(list(ty.shape))):
                tile[dim.position] = size
            i64 = ir.IntegerType.get_signless(64)
            results.set_params(op.wg_tile, [ir.IntegerAttr.get(i64, t) for t in tile])
            results.set_params(op.sg_tile, [])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "InferXeGPUElementwiseParamsOp",
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


def infer_xegpu_elementwise_params(
    target: ir.Value[transform.AnyOpType],
) -> tuple[ir.Value, ir.Value]:
    """
    snake_case wrapper to create an InferXeGPUElementwiseParamsOp.

    Args:
        target: Handle to a single tensor elementwise or vector memory anchor.
    Return:
        Workgroup and subgroup tile params. The subgroup tile is empty for
        linalg anchors.
    """
    op = InferXeGPUElementwiseParamsOp(target=target)
    return op.wg_tile, op.sg_tile

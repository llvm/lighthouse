from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa


class GetTileSizesOp(TransformExtensionDialect.Operation, name="get_tile_sizes"):
    """
    Read the tile sizes annotated on an op as a transform param.

    Reads the annotation of a single op and returns its entries as a param,
    one integer per iteration dimension (loop order).

    `target` must resolve to a single annotated op.

    Args:
        target: Handle to a single annotated op.
    Return:
        Param holding the op's tile sizes. Empty if the op is not annotated.
    """

    target: ext.Operand[transform.AnyOpType]
    tile_sizes_param: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def _size_attr(value: int) -> ir.IntegerAttr:
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            sizes = tsa.get_tile_sizes_attr(target_ops[0])
            if sizes is None:
                sizes = []

            params = [GetTileSizesOp._size_attr(size) for size in sizes]
            results.set_params(op.tile_sizes_param, params)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_tile_sizes(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a GetTileSizesOp.

    Args:
        target: Handle to a single annotated op.
    Returns:
        Param holding the op's tile sizes.
    """
    return GetTileSizesOp(target=target).tile_sizes_param

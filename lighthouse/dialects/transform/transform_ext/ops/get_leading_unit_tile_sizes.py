from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import num_loops


class GetLeadingUnitTileSizesOp(
    TransformExtensionDialect.Operation, name="get_leading_unit_tile_sizes"
):
    """
    Return unit tile sizes for all but the innermost loop of an op.

    For an op with N iteration dims, returns a param of N-1 ones. Tiling by these
    turns the leading dims into loops and leaves the innermost dim whole, which is
    a rank-agnostic way to prepare an op for innermost-dimension vectorization.

    Args:
        target: Handle to a single structured linalg op.
    Return:
        Param of N-1 unit tile sizes (empty for a 0/1-dim op).
    """

    target: ext.Operand[transform.AnyOpType]
    tile_sizes_param: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetLeadingUnitTileSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            loops = num_loops(target_ops[0])
            if loops is None:
                return DiagnosedSilenceableFailure.SilenceableFailure

            i64 = ir.IntegerType.get_signless(64)
            params = [ir.IntegerAttr.get(i64, 1) for _ in range(max(0, loops - 1))]
            results.set_params(op.tile_sizes_param, params)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetLeadingUnitTileSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_leading_unit_tile_sizes(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a GetLeadingUnitTileSizesOp.

    Args:
        target: Handle to a single structured linalg op.
    Returns:
        Param of N-1 unit tile sizes for an N-dim op.
    """
    return GetLeadingUnitTileSizesOp(target=target).tile_sizes_param

from typing import Sequence, Optional

from mlir import ir
from mlir.dialects import ext, transform, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GetTilingSizesOp(TransformExtensionDialect.Operation, name="get_tiling_sizes"):
    """
    Get tile sizes for given operation.

    The analysis targets high-level tiling aiming to expose parallelism
    and improve memory access patterns.
    Only parallel dimensions are tiled.

    Currently, only Linalg op are supported.
    The target must be a single operation.

    Args:
        target: Handle to target.
        tile_dim: Optional size used for tile dimensions (default: 32).
            TODO: Allow more fine-grained control.
    Return:
        Handle holding tile size values.
        Returns empty handle if no analysis is available for the given op.
    """

    target: ext.Operand[transform.AnyOpType]
    tile_dim: Optional[ext.Operand[transform.AnyParamType]] = None
    tile_sizes_param: ext.Result[transform.AnyParamType[()]] = ext.result(
        infer_type=True
    )

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @classmethod
    def tile_param_attr(cls, value: int):
        """Get attribute suitable for use as a tiling size."""
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)

    @staticmethod
    def named_op_matmul_tiles(named_op: ir.OpView, tile_size: int) -> Sequence[int]:
        """
        Get tiling sizes for Linalg matmul named ops variants.

        Args:
            named_op: Target named op.
            tile_size: Size of all tile dimensions.
        Returns:
            List of tile sizes for all op's iterators.
            Empty if the op is not supported.
        """
        c_type: ir.ShapedType = named_op.outputs[0].type
        if any(
            ir.ShapedType.is_static_size(dim) and dim < tile_size
            for dim in c_type.shape
        ):
            # Disable tiling.
            tile_size = 0

        match named_op:
            case linalg.MatmulOp():
                return [tile_size, tile_size, 0]
            case linalg.BatchMatmulOp():
                batch_tile = 1 if tile_size != 0 else 0
                return [batch_tile, tile_size, tile_size, 0]
            case linalg.BatchReduceMatmulOp():
                return [0, tile_size, tile_size, 0]
            case _:
                return []

    @staticmethod
    def contract_tiles(contract: linalg.ContractOp, tile_size: int) -> Sequence[int]:
        """
        Get tiling sizes for Linalg contraction op.

        Args:
            contract: Target contract op.
            tile_size: Size of all tile dimensions.
        Returns:
            List of tile sizes for all op's iterators.
        """
        c_map: ir.AffineMap = contract.indexing_maps[2].value
        c_type: ir.ShapedType = contract.outputs[0].type
        num_tile_dims = min(2, c_type.rank)
        if any(
            ir.ShapedType.is_static_size(dim) and dim < tile_size
            for dim in c_type.shape[-num_tile_dims:]
        ):
            # No tiling.
            # Return fixed number of handles for predictable
            # usage with handle splitting.
            return [0] * c_map.n_dims

        # Create up to 2D tiles on the innermost dimensions.
        # All other parallel dimensions are tiled into unit dimensions.
        tile_sizes = [0] * c_map.n_dims
        for dim in c_map.results[:-num_tile_dims]:
            tile_sizes[dim.position] = 1
        for dim in c_map.results[-num_tile_dims:]:
            tile_sizes[dim.position] = tile_size
        return tile_sizes

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetTilingSizesOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            tile_size = 32
            if op.tile_dim is not None:
                tile_attr = state.get_params(op.tile_dim)
                if len(tile_attr) == 1 and isinstance(tile_attr[0], ir.IntegerAttr):
                    tile_size = tile_attr[0].value

            target = target_ops[0].opview
            match target:
                case (
                    linalg.MatmulOp()
                    | linalg.BatchMatmulOp()
                    | linalg.BatchReduceMatmulOp()
                ):
                    tile_sizes = op.named_op_matmul_tiles(target, tile_size=tile_size)
                case linalg.ContractOp():
                    tile_sizes = op.contract_tiles(target, tile_size=tile_size)
                case _:
                    tile_sizes = []

            tile_params = [op.tile_param_attr(tile) for tile in tile_sizes]
            results.set_params(op.tile_sizes_param, tile_params)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetTilingSizesOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_tiling_sizes(
    target: ir.Value[transform.AnyOpType],
    tile_dim: int | ir.Value | None = None,
) -> ir.Value:
    """
    snake_case wrapper to create a GetTilingSizesOp.

    Args:
        target: Handle to target op
        tile_dim: Optional size used for tile dimensions (default: 32)
    Returns:
        Created op
    """
    if isinstance(tile_dim, int):
        param_attr = GetTilingSizesOp.tile_param_attr(tile_dim)
        tile_dim = transform.ParamConstantOp(transform.AnyParamType.get(), param_attr)

    return GetTilingSizesOp(target=target, tile_dim=tile_dim)

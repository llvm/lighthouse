from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.dialects.transform.transform_ext.utils import tile_size_analysis as tsa
from lighthouse.dialects.transform.transform_ext.utils import fusion_analysis as fa


class ClearTileAndFuseAnnotationsOp(
    TransformExtensionDialect.Operation, name="clear_tile_and_fuse_annotations"
):
    """
    Remove the tile-and-fuse annotations from the target ops and their nested ops.

    Each target op is walked, so passing a container handle (e.g. the `scf.for` loops
    produced by fusion) clears the annotations of every op inside it.

    Ops without the annotations are left untouched. The original `target` handle
    is returned for chaining.

    Args:
        target: Handle to op(s) whose subtree(s) to clear.
    Return:
        Pass-through handle to the target ops.
    """

    target: ext.Operand[transform.AnyOpType]
    cleared: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def _clear(visited: ir.Operation) -> ir.WalkResult:
        tsa.clear_tile_sizes_attr(visited)
        fa.clear_fusion_boundary(visited)
        return ir.WalkResult.ADVANCE

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ClearTileAndFuseAnnotationsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = list(state.get_payload_ops(op.target))
            for target_op in target_ops:
                # Walk the whole subtree so a container handle (e.g. a fused
                # loop) clears the annotations of the ops nested inside it.
                target_op.walk(
                    ClearTileAndFuseAnnotationsOp._clear, ir.WalkOrder.PRE_ORDER
                )

            results.set_ops(op.cleared, target_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "ClearTileAndFuseAnnotationsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def clear_tile_and_fuse_annotations(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a ClearTileAndFuseAnnotationsOp.

    Args:
        target: Handle to op(s) to clear.
    Returns:
        Pass-through handle to the target ops.
    """
    return ClearTileAndFuseAnnotationsOp(target=target).cleared

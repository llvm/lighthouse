from mlir import ir
from mlir.dialects import ext, transform, linalg, cf
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class RemoveAssertsOp(TransformExtensionDialect.Operation, name="remove_asserts"):
    """Remove asserts within Linalg op bodies.

    A workaround to enable vectorization of Linalg ops by removing
    internal asserts.
    """

    target: ext.Operand[transform.AnyOpType]
    updated_op: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            target_ops = list(state.get_payload_ops(op.target))

            new_ops = []
            for target_op in target_ops:
                if not isinstance(target_op.opview, linalg.GenericOp):
                    continue
                updated = False
                for body_op in target_op.opview.region.blocks[0].operations:
                    if isinstance(body_op, cf.AssertOp):
                        rewriter.erase_op(body_op)
                        updated = True
                if updated:
                    new_ops.append(target_op)

            results.set_ops(op.updated_op, new_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.only_reads_handle(op.op_operands[:1])
                + transform.consumes_handle(op.op_operands[1:])
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def remove_asserts(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """snake_case wrapper to create a RemoveAssertsOp."""
    return RemoveAssertsOp(target=target).updated_op

from mlir import ir
from mlir.dialects import ext, transform, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GetTileableConsumersOp(
    TransformExtensionDialect.Operation, name="get_tileable_consumers"
):
    """
    Find consumer ops of the `target` operation that are tileable linalg ops.

    If no such consumers are found, the operation returns the target itself.

    Args:
        target: Handle to target op
    Returns:
        List of tileable consumer ops, or the target op itself.
    """

    target: ext.Operand[transform.AnyOpType]
    ops: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    @staticmethod
    def get_op_users(val: ir.Value) -> list[ir.Operation]:
        op_users = []
        for use in val.uses:
            user = use.owner
            if not isinstance(user, ir.OpView):
                continue
            op_users.append(user.operation)
        return op_users

    @staticmethod
    def is_tileable_op(op: ir.Operation) -> bool:
        # TODO expand list as needed and/or check traits/interfaces
        linalg_ops = [
            linalg.ElementwiseOp,
            linalg.AddOp,
            linalg.SubOp,
            linalg.MulOp,
            linalg.DivOp,
            linalg.ExpOp,
            linalg.MaxOp,
            linalg.MinOp,
            linalg.FillOp,
            linalg.MatmulOp,
            linalg.GenericOp,
        ]
        return isinstance(op.opview, tuple(linalg_ops))

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetTileableConsumersOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)

            if len(target_ops) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure

            new_ops = []
            target: ir.Operation = target_ops[0]
            op_res = target.results
            while len(op_res) == 1:
                users = op.get_op_users(op_res[0])
                if len(users) != 1:
                    break
                user = users[0]
                if not op.is_tileable_op(user):
                    break
                new_ops.append(user)
                op_res = user.results

            if not new_ops:
                new_ops = [target]
            results.set_ops(op.ops, new_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetTileableConsumersOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_tileable_consumers(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value:
    """
    snake_case wrapper to create a GetTileableConsumersOp.

    Args:
        target: Handle to target op
    Returns:
        List of tileable consumer ops, or the target op itself.
    """
    return GetTileableConsumersOp(target=target).ops

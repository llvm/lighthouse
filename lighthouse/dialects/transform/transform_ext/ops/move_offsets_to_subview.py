from mlir import ir
from mlir.dialects import ext, transform, vector, memref, arith
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def get_base(op: ir.OpView) -> ir.Value:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.base


def get_vector_type(op: ir.OpView) -> ir.VectorType:
    if isinstance(op, vector.TransferReadOp):
        return op.vector.type
    elif isinstance(op, vector.TransferWriteOp):
        return op.valueToStore.type
    else:
        raise NotImplementedError("Unsupported op")


def get_indices(op: ir.OpView) -> ir.OpOperandList:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.indices


def get_permutation_map(op: ir.OpView) -> ir.AffineMapAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.permutation_map


def get_in_bounds(op: ir.OpView) -> ir.ArrayAttr:
    assert isinstance(op, (vector.TransferReadOp, vector.TransferWriteOp)), (
        "Expected vector transfer op"
    )
    return op.in_bounds


class MoveOffsetsToSubviewOp(
    TransformExtensionDialect.Operation, name="move_offsets_to_subview"
):
    target: ext.Operand[transform.AnyOpType]
    updated_op: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def create_subview(target: ir.OpView) -> ir.OpView | None:
        assert isinstance(target, (vector.TransferReadOp, vector.TransferWriteOp)), (
            "Expected vector transfer op"
        )

        base_val = get_base(target)
        assert isinstance(base_val.type, ir.MemRefType), (
            "Expected memref type for base value"
        )

        memref_type: ir.MemRefType = base_val.type
        vec_type: ir.VectorType = get_vector_type(target)

        map_attr = get_permutation_map(target)
        map: ir.AffineMap = map_attr.value
        if map != ir.AffineMap.get_minor_identity(map.n_dims, len(map.results)):
            return None

        sizes = memref_type.shape
        strides = [1] * memref_type.rank
        indices = get_indices(target)

        subview_offsets = []
        zero_offset = arith.ConstantOp(ir.IndexType.get(), 0)
        for offset in indices:
            if not isinstance(offset.owner, arith.ConstantOp):
                subview_offsets.append(offset)
            else:
                subview_offsets.append(zero_offset)

        base_subview = memref.subview(
            base_val,
            subview_offsets,
            sizes,
            strides,
        )

        transfer_indices = []
        for offset in indices:
            if isinstance(offset.owner, arith.ConstantOp):
                transfer_indices.append(offset)
            else:
                transfer_indices.append(zero_offset)
        transfer_op = (
            vector.transfer_read(
                vec_type,
                base_subview,
                transfer_indices,
                map_attr,
                target.padding,
                get_in_bounds(target),
            ).owner
            if isinstance(target, vector.TransferReadOp)
            else vector.transfer_write(
                None,
                target.valueToStore,
                base_subview,
                transfer_indices,
                map_attr,
                get_in_bounds(target),
            )
        )

        return transfer_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "MoveOffsetsToSubviewOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            updated_ops = []

            for target in targets:
                if not isinstance(
                    target, (vector.TransferReadOp, vector.TransferWriteOp)
                ):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                base_val = get_base(target)
                if not isinstance(base_val.type, ir.MemRefType):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                with ir.InsertionPoint(target), target.location:
                    updated_op = MoveOffsetsToSubviewOp.create_subview(target)
                    if updated_op is None:
                        updated_op = target
                    else:
                        rewriter.replace_op(target, updated_op)
                updated_ops.append(updated_op)

            results.set_ops(op.updated_op, updated_ops)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "MoveOffsetsToSubviewOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "MoveOffsetsToSubviewOp", effects):
            transform.consumes_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def move_offsets_to_subview(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a MoveOffsetsToSubviewOp."""
    op = MoveOffsetsToSubviewOp(target=target)
    return op.updated_op

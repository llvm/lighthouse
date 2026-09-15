from mlir import ir
from mlir.dialects import ext, transform, memref
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class UpdateAddressSpaceOp(
    TransformExtensionDialect.Operation, name="update_address_space"
):
    """Update the address space of a memref allocation operation.

    Takes a target memref allocation operation and updates its address space
    to the provided value.
    """

    target: ext.Operand[transform.AnyOpType]
    address_space: ir.IntegerAttr
    updated_op: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "UpdateAddressSpaceOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            # Get the target operations to transform
            target_ops = state.get_payload_ops(op.target)
            # Get the address space value from the attribute
            address_space_value = ir.IntegerAttr(op.address_space).value
            new_ops = []
            for target_op in target_ops:
                # Verify this is a memref.alloca operation
                if not isinstance(target_op.opview, memref.AllocaOp):
                    return DiagnosedSilenceableFailure.emit_silenceable_error(
                        f"Expected memref.alloca operation, got {target_op.OPERATION_NAME}"
                    )

                # Get the current result type (should be a MemRefType)
                old_result_type = target_op.results[0].type
                memref_type = ir.MemRefType(old_result_type)
                # Create a new memref type with the specified address space
                new_memref_type = ir.MemRefType.get(
                    memref_type.shape,
                    memref_type.element_type,
                    layout=memref_type.layout,
                    memory_space=ir.Attribute.parse(str(address_space_value)),
                )

                # Replace the operation with a new one that has the updated type
                with ir.InsertionPoint(target_op):
                    # Get the operands from the original alloca (dynamic sizes and symbols)
                    dynamic_sizes = list(
                        target_op.operands[
                            : target_op.attributes["operandSegmentSizes"][0]
                        ]
                    )
                    symbol_operands = list(
                        target_op.operands[
                            target_op.attributes["operandSegmentSizes"][0] :
                        ]
                    )
                    # Create a new alloca with the updated type
                    new_alloca = memref.alloca(
                        new_memref_type, dynamic_sizes, symbol_operands
                    )
                    # Replace all uses of the old operation with the new one
                    # FIXME: This won't handle operations that consume the memref type and
                    # return a new memref (such as subview).
                    rewriter.replace_op(target_op, [new_alloca])
                    new_ops.append(new_alloca.owner)

            # Set the results to the new operations
            results.set_ops(op.updated_op, new_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "UpdateAddressSpaceOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                transform.consumes_handle(op.op_operands[:1])
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def update_address_space(
    target: ir.Value,
    address_space: int | ir.IntegerAttr,
) -> ir.Value:
    if not isinstance(address_space, ir.IntegerAttr):
        address_space = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), address_space
        )
    return UpdateAddressSpaceOp(target, address_space=address_space).updated_op

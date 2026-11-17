"""Expand boolean-to-float casts to avoid an LLVM SPIR-V backend bug."""

from mlir import ir
from mlir.dialects import arith, ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ExpandBoolToFloatOp(
    TransformExtensionDialect.Operation, name="expand_bool_to_float"
):
    """Replace scalar/vector unsigned i1-to-float casts with floating selects.

    LLVM's SPIR-V instruction selector can crash on vector boolean uitofp op.
    Converting the `arith.uitofp` op to `arith.select` that selects
    floating-point one or zero avoids the issue and preserves cast semantics.

    Apply before XeGPU distribution and binary lowering. Signed casts, wider
    integers and tensor casts are left unchanged.

    Consumes the container handle and invalidates nested handles. Returns the
    same container so subsequent transforms can reacquire payload handles.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state) -> DiagnosedSilenceableFailure:
            targets = [target.operation for target in state.get_payload_ops(op.target)]
            if len(targets) != 1:
                op.operation.location.emit_error(
                    "expand_bool_to_float expects one container"
                )
                return DiagnosedSilenceableFailure.SilenceableFailure
            casts = []

            def visit(operation):
                if isinstance(operation.opview, arith.UIToFPOp):
                    casts.append(operation.opview)
                return ir.WalkResult.ADVANCE

            targets[0].walk(visit)
            for cast in casts:
                source_type = cast.operands[0].type
                result_type = cast.results[0].type
                if isinstance(source_type, ir.VectorType):
                    source_element = source_type.element_type
                    result_element = ir.VectorType(result_type).element_type
                elif isinstance(source_type, ir.IntegerType):
                    source_element = source_type
                    result_element = result_type
                else:
                    continue
                if not isinstance(source_element, ir.IntegerType):
                    continue
                if ir.IntegerType(source_element).width != 1:
                    continue
                with ir.InsertionPoint(cast), cast.location:
                    values = []
                    for value in (1.0, 0.0):
                        scalar = ir.FloatAttr.get(result_element, value)
                        if isinstance(result_type, ir.VectorType):
                            constant = ir.DenseElementsAttr.get_splat(
                                result_type, scalar
                            )
                        else:
                            constant = scalar
                        values.append(arith.ConstantOp(result_type, constant).result)
                    select = arith.SelectOp(cast.operands[0], *values)
                rewriter.replace_op(cast, [select.result])
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def expand_bool_to_float(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """Expand unsigned scalar/vector boolean-to-float casts into selects."""
    return ExpandBoolToFloatOp(target=target).result

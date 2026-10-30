from mlir import ir
from mlir.dialects import arith, ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class FoldLogOfBoolOp(TransformExtensionDialect.Operation, name="fold_log_of_bool"):
    """Replace log(uitofp(i1)) with select(predicate, +0.0, -inf).

    Required due to our torch-mlir causal masking workaround: importing SDPA
    with a boolean or implicit causal mask fails, so Torch constructs an
    additive mask using allowed.to(dtype).log(). This lowers to a `math.log` op
    on a boolean tensor, which is expensive to evaluate. This transform
    replaces the log of a boolean-to-float cast with a select of +0.0 and -inf,
    which is much cheaper to evaluate.

    The rewrite is generic, not causal-mask-specific. Matches only direct
    unsigned i1-to-float casts, scalar or vector. Signed casts, wider integers,
    arbitrary floating inputs and tensor casts are unchanged. Shared casts
    remain intact for other users. No fast-math assumptions are needed.

    Consumes container handles and invalidates nested handles; returns the
    same containers so subsequent transforms can reacquire payload handles.
    """

    target: ext.Operand[transform.AnyOpType]
    result: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(op, rewriter, results, state):
            targets = list(state.get_payload_ops(op.target))
            logs = []

            def visit(inner):
                if inner.name == "math.log":
                    logs.append(inner)
                return ir.WalkResult.ADVANCE

            for target in targets:
                target.walk(visit)
            for log in logs:
                source = log.operands[0]
                if isinstance(source.owner, ir.Block):
                    continue
                cast = source.owner.operation
                if cast.name != "arith.uitofp":
                    continue
                predicate = cast.operands[0]
                source_type = predicate.type
                result_type = source.type
                if isinstance(source_type, ir.VectorType):
                    source_element = source_type.element_type
                    result_element = ir.VectorType(result_type).element_type
                elif isinstance(source_type, ir.IntegerType):
                    source_element = source_type
                    result_element = result_type
                else:
                    continue
                if (
                    not isinstance(source_element, ir.IntegerType)
                    or source_element.width != 1
                ):
                    continue
                with ir.InsertionPoint(log), log.location:
                    values = []
                    for value in (0.0, float("-inf")):
                        constant = ir.FloatAttr.get(result_element, value)
                        if isinstance(result_type, ir.VectorType):
                            constant = ir.DenseElementsAttr.get_splat(
                                result_type, constant
                            )
                        values.append(arith.ConstantOp(result_type, constant).result)
                    select = arith.SelectOp(predicate, *values)
                rewriter.replace_op(log, [select.result])
            results.set_ops(op.result, targets)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op):
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op):
            return (
                transform.consumes_handle(op.op_operands)
                + transform.produces_handle(op.results)
                + transform.modifies_payload()
            )


def fold_log_of_bool(target: ir.Value) -> ir.Value:
    """Fold scalar/vector log of an unsigned boolean-to-float cast."""
    return FoldLogOfBoolOp(target=target).result

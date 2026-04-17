from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class GetNamedAttributeOp(
    TransformExtensionDialect.Operation, name="get_named_attribute"
):
    """
    Obtain a `target` op's associated attribute by `attr_name` as a `param`.

    In case `target` resolves to multiple ops, the operation returns a list of
    attributes. If any of the resolved `target` ops does not have an attribute
    with the name `attr_name`, the operation fails.
    """

    target: ext.Operand[transform.AnyOpType]
    attr_name: ir.StringAttr
    param: ext.Result[transform.AnyParamType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "GetNamedAttributeOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)
            associated_attrs = []
            for target_op in target_ops:
                assoc_attr = target_op.attributes.get(op.attr_name.value)
                if assoc_attr is None:
                    return DiagnosedSilenceableFailure.SilenceableFailure
                associated_attrs.append(assoc_attr)
            results.set_params(op.param, associated_attrs)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "GetNamedAttributeOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


def get_named_attribute(
    target: ir.Value, attr_name: str | ir.StringAttr
) -> ir.Value[transform.AnyParamType]:
    if not isinstance(attr_name, ir.StringAttr):
        attr_name = ir.StringAttr.get(attr_name)
    return GetNamedAttributeOp(target=target, attr_name=attr_name).param

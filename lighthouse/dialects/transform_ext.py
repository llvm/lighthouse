from typing import Sequence

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure


def register_and_load(**kwargs):
    TransformDialectExtension.load(**kwargs)


@ext.register_dialect
class TransformDialectExtension(ext.Dialect, name="transform_ext"):
    @classmethod
    def load(cls, *args, **kwargs):
        super(TransformDialectExtension, cls).load(*args, **kwargs, register=False)

        for op in cls.operations:
            if hasattr(op, "attach_interfaces"):
                op.attach_interfaces()


@ext.register_operation(TransformDialectExtension)
class GetNamedAttributeOp(
    TransformDialectExtension.Operation, name="get_named_attribute"
):
    param: ext.Result[transform.AnyParamType[()]]
    target: ext.Operand[transform.AnyOpType]
    attr_name: ir.StringAttr

    @classmethod
    def attach_interfaces(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

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


def get_named_attribute(target, attr_name) -> ir.Value[transform.AnyParamType]:
    if not isinstance(attr_name, ir.StringAttr):
        attr_name = ir.StringAttr.get(attr_name)
    return GetNamedAttributeOp(target=target, attr_name=attr_name).param


@ext.register_operation(TransformDialectExtension)
class ParamCmpEqOp(TransformDialectExtension.Operation, name="param_cmp_eq"):
    lhs: ext.Operand[transform.AnyParamType]
    rhs: ext.Operand[transform.AnyParamType]

    @classmethod
    def attach_interfaces(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ParamCmpEqOp",
            _rewriter: transform.TransformRewriter,
            _results: transform.TransformResults,
            state: transform.TransformState,
        ) -> transform.DiagnosedSilenceableFailure:
            lhs_params = state.get_params(op.lhs)
            rhs_params = state.get_params(op.rhs)
            if len(lhs_params) != len(rhs_params):
                return transform.DiagnosedSilenceableFailure.SilenceableFailure
            for lhs_param, rhs_param in zip(lhs_params, rhs_params):
                if lhs_param != rhs_param:
                    return transform.DiagnosedSilenceableFailure.SilenceableFailure
            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ParamCmpEqOp") -> bool:
            return True

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ParamCmpEqOp", effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.only_reads_payload(effects)


def param_cmp_eq(lhs, rhs):
    return ParamCmpEqOp(lhs=lhs, rhs=rhs)


@ext.register_operation(TransformDialectExtension)
class ReplaceOp(TransformDialectExtension.Operation, name="replace"):
    new_op: ext.Result[transform.AnyOpType[()]]
    target: ext.Operand[transform.AnyOpType]
    new_name: ir.StringAttr
    new_operands: Sequence[ext.Operand[transform.AnyValueType]]

    @classmethod
    def attach_interfaces(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ReplaceOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            target_ops = state.get_payload_ops(op.target)

            # Resolve optional operand handles to payload values.
            operand_values_per_handle = []
            for operand_handle in op.new_operands:
                operand_values_per_handle.append(
                    state.get_payload_values(operand_handle)
                )
                assert len(operand_values_per_handle[-1]) == len(target_ops), (
                    "Expected number of operand values to match number of target ops"
                )

            new_op_name = op.new_name.value
            new_op_attrs = {}
            if "new_attrs" in op.attributes:
                new_attrs = op.attributes["new_attrs"]
                assert isinstance(new_attrs, ir.DictAttr), (
                    "Expected new_attrs to be a dictionary attribute"
                )
                for named_attr in new_attrs:
                    new_op_attrs[named_attr.name] = named_attr.attr

            new_ops = []
            for target_idx, target_op in enumerate(target_ops):
                if "new_result_types" in op.attributes:
                    tuple_type = op.attributes["new_result_types"].value
                    assert isinstance(tuple_type, ir.TupleType), (
                        "Expected new_result_types to be a tuple of types"
                    )
                    assert tuple_type.num_types == len(target_op.results), (
                        "Expected number of new result types to match number of target op results"
                    )

                    new_result_types = [
                        tuple_type.get_type(i) for i in range(tuple_type.num_types)
                    ]
                else:
                    new_result_types = [ty.type for ty in target_op.results]

                if operand_values_per_handle:
                    new_operands = [
                        vals[target_idx] for vals in operand_values_per_handle
                    ]
                else:
                    new_operands = list(target_op.operands)

                with ir.InsertionPoint(target_op):
                    new_operation = ir.Operation.create(
                        new_op_name,
                        results=new_result_types,
                        operands=new_operands,
                        attributes=new_op_attrs,
                    )
                    rewriter.replace_op(target_op, new_operation)
                    new_ops.append(new_operation)

            results.set_ops(op.new_op, new_ops)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ReplaceOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation, effects):
            transform.consumes_handle(op.op_operands[:1], effects)
            if new_operands_handles := op.op_operands[1:]:
                transform.only_reads_handle(new_operands_handles, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def replace(
    target, new_name, *new_operands, new_result_types=None, new_attrs=None
) -> ir.Value:
    if not isinstance(new_name, ir.StringAttr):
        new_name = ir.StringAttr.get(new_name)
    new_op = ReplaceOp(target, new_name=new_name, new_operands=new_operands).new_op
    if new_result_types:
        if not isinstance(new_result_types, ir.TupleType):
            new_result_types = ir.TupleType.get_tuple(new_result_types)
        new_op.owner.attributes["new_result_types"] = ir.TypeAttr.get(new_result_types)
    if new_attrs:
        if isinstance(new_attrs, dict):
            new_attrs = ir.DictAttr.get(new_attrs)
        else:
            assert isinstance(new_attrs, ir.DictAttr)
        new_op.owner.attributes["new_attrs"] = new_attrs
    return new_op

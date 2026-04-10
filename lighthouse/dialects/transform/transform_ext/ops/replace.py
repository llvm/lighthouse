from typing import Sequence, Optional

from mlir import ir
from mlir.dialects import ext, transform
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


class ReplaceOp(TransformExtensionDialect.Operation, name="replace"):
    """Replace the `target` operation(s) with a new `op_kind` operation.

    If `new_operands` are provided, they are used as operands for the new
    operation(s); otherwise, the operands of the `target` operation(s) are
    reused. The new op's result types are the same as those of the `target` op.

    NB: This op is mostly an escape hatch for testing and prototyping purposes.
    No attempt is made to guarantee that the rewrite is semantics perserving.
    """

    target: ext.Operand[transform.AnyOpType]
    op_kind: ir.StringAttr
    new_operands: Sequence[ext.Operand[transform.AnyValueType]]
    new_op: ext.Result[transform.AnyOpType[()]] = ext.result(infer_type=True)

    @classmethod
    def attach_interface_impls(cls, ctx=None):
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

            new_op_name = op.op_kind.value
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
    target: ir.Value,
    op_kind: str | ir.StringAttr,
    *new_operands: ir.Value,
    new_result_types: Optional[ir.TupleType | Sequence[ir.Type]] = None,
    new_attrs=None,
) -> ir.Value:
    if not isinstance(op_kind, ir.StringAttr):
        op_kind = ir.StringAttr.get(op_kind)
    new_op = ReplaceOp(target, op_kind=op_kind, new_operands=new_operands).new_op
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

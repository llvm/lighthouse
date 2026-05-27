from mlir import ir
from mlir.dialects import ext, transform, func, arith, linalg
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.utils.mlir import func_cif
import numpy as np


class ConvertConstResourcesToArgsOp(
    TransformExtensionDialect.Operation, name="convert_const_resources_to_args"
):
    """Converts all arith.constant dense_resource ops to function arguments.

    Dense resource const ops are detected and converted into function
    arguments. New function arguments are placed at the end of the argument
    list. The arguments are ordered by matmul layers and their role in the
    matmul's A, B, or epilogue compute chain, e.g. matmul_0_A, matmul_0_B,
    matmul_0_epilogue, matmul_1_A, etc.

    Currently supports only functions with tensor arguments and return values.
    """

    target: ext.Operand[transform.AnyOpType]
    converted_func: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def convert_func(target: func.FuncOp) -> func.FuncOp:
        func_name = target.sym_name.value
        func_inputs = list(target.type.inputs)
        func_results = list(target.type.results)
        assert all(isinstance(ty, ir.RankedTensorType) for ty in func_inputs), (
            "Only tensors are supported as input types"
        )
        assert all(isinstance(ty, ir.RankedTensorType) for ty in func_results), (
            "Only tensors are supported as return types"
        )

        def is_const_resource(op: ir.Operation) -> bool:
            if isinstance(op, arith.ConstantOp):
                cst = op.value
                cst_type = cst.type
                if (
                    isinstance(cst_type, ir.RankedTensorType)
                    and np.prod(cst_type.shape) > 1
                    and not isinstance(cst, ir.DenseElementsAttr)
                ):
                    # a tensor with > 1 element and not dense constant e.g. dense<0.0>
                    return True
            return False

        def trace_producers(val: ir.Value) -> ir.Operation | None:
            """Trace producers to find a const resource."""
            if val is None:
                return None
            if isinstance(val, ir.BlockArgument):
                return None
            if isinstance(val, ir.OpResult):
                parent_op = val.owner
                if is_const_resource(parent_op):
                    return parent_op
                if isinstance(parent_op, linalg.MatmulOp):
                    # stop trace at matmul
                    return None
                # recursively check producers
                for operand in parent_op.operands:
                    found = trace_producers(operand)
                    if found is not None:
                        return found
            return None

        def trace_consumers(val: ir.Value) -> ir.Operation | None:
            """Trace consumers to find if any of them depends on a const resource."""
            if val is None:
                return None
            for use in val.uses:
                user = use.owner
                if isinstance(user, linalg.MatmulOp):
                    # stop trace at next matmul
                    return None
                # check if any other operand can be traced back to a const resource
                for operand in user.operands:
                    if operand == val:
                        continue
                    found = trace_producers(operand)
                    if found is not None:
                        return found
                for result in user.results:
                    found = trace_consumers(result)
                    if found is not None:
                        return found
            return None

        const_ops = []

        def find_matmul_resources(op: ir.Operation) -> ir.WalkResult:
            """Find matmul ops, trace producers/consumers to find const resources"""
            op = op.opview
            if isinstance(op, linalg.MatmulOp):
                a_const = trace_producers(op.inputs[0])
                b_const = trace_producers(op.inputs[1])
                if a_const is not None and a_const not in const_ops:
                    const_ops.append(a_const)
                if b_const is not None and b_const not in const_ops:
                    const_ops.append(b_const)
                use_chain_const = trace_consumers(op.results[0])
                if use_chain_const is not None and use_chain_const not in const_ops:
                    const_ops.append(use_chain_const)
            return ir.WalkResult.ADVANCE

        def find_other_const_resources(op: ir.Operation) -> ir.WalkResult:
            op = op.opview
            if is_const_resource(op) and op not in const_ops:
                raise NotImplementedError(
                    f"Found a const resource that is not connected to a matmul: \n {op}"
                )
            return ir.WalkResult.ADVANCE

        target.walk(find_matmul_resources, ir.WalkOrder.PRE_ORDER)
        target.walk(find_other_const_resources, ir.WalkOrder.PRE_ORDER)

        new_inputs = [cst.value.type for cst in const_ops]
        new_args = [ty for ty in func_inputs + new_inputs]

        @func_cif(*new_args, name=func_name)
        def f(*args):
            const_to_arg = {k: v for k, v in zip(const_ops, args[len(func_inputs) :])}

            # keep track of cloned ops to replace operands
            cloned_op_to_op = {}
            cloned_op_to_value = {}

            def replace_operands(op: ir.Operation) -> ir.WalkResult:
                for i, oprnd in enumerate(op.operands):
                    if isinstance(oprnd, ir.BlockArgument):
                        # operand is block argument
                        owner = oprnd.owner.owner  # op that owns the block
                        if owner == target:
                            # payload func argument, replace with input tensor
                            op.operands[i] = args[oprnd.arg_number]
                    else:
                        # replace operands with cloned values
                        if oprnd.owner in cloned_op_to_value:
                            op.operands[i] = cloned_op_to_value[oprnd.owner]
                        elif oprnd.owner in cloned_op_to_op:
                            res_idx = oprnd.result_number
                            op.operands[i] = cloned_op_to_op[oprnd.owner].results[
                                res_idx
                            ]
                return ir.WalkResult.ADVANCE

            for op in target.regions[0].blocks[0].operations:
                if isinstance(op, func.ReturnOp):
                    # return the new values
                    new_return_values = []
                    for res_val in op.operands:
                        if res_val.owner not in cloned_op_to_op:
                            raise NotImplementedError("Unsupported return value")
                        res_idx = res_val.result_number
                        new_val = cloned_op_to_op[res_val.owner].results[res_idx]
                        new_return_values.append(new_val)
                    return new_return_values
                else:
                    if is_const_resource(op.opview):
                        # use new func arg value instead
                        cloned_op_to_value[op] = const_to_arg[op]
                    else:
                        # clone all other ops
                        new_op = op.clone()
                        new_op.walk(replace_operands, ir.WalkOrder.PRE_ORDER)
                        cloned_op_to_op[op] = new_op

        return f.func_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ConvertConstResourcesToArgsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            converted_funcs = []

            for target in targets:
                if not isinstance(target, func.FuncOp):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                with ir.InsertionPoint(target), target.location:
                    new_func = ConvertConstResourcesToArgsOp.convert_func(target)
                    target.erase()
                converted_funcs.append(new_func)

            results.set_ops(op.converted_func, converted_funcs)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(
            _op: "ConvertConstResourcesToArgsOp",
        ) -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ConvertConstResourcesToArgsOp", effects):
            transform.consumes_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def convert_const_resources_to_args(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a ConvertConstResourcesToArgsOp."""
    op = ConvertConstResourcesToArgsOp(target=target)
    return op.converted_func

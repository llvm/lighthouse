from mlir import ir
from mlir.dialects import ext, transform, func, bufferization
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor
from lighthouse.utils.mlir import func_cif


class ConvertFuncResultsToArgsOp(
    TransformExtensionDialect.Operation, name="convert_func_results_to_args"
):
    """Converts all function return values to function arguments.

    Function return values are placed in the beginning of the argument list,
    followed by the original function arguments.

    Function arguments are converted to memrefs with appropriate bufferization
    annotations for inputs (bufferization.to_tensor with restrict=True) and
    outputs (bufferization.materialize_in_destination).

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
        def memref_t(ttype: ir.Type) -> ir.MemRefType:
            return ir.MemRefType.get(ttype.shape, ttype.element_type)

        func_name = target.sym_name.value
        func_inputs = list(target.type.inputs)
        func_results = list(target.type.results)
        assert all(isinstance(ty, ir.RankedTensorType) for ty in func_inputs), (
            "Only tensors are supported as input types"
        )
        assert all(isinstance(ty, ir.RankedTensorType) for ty in func_results), (
            "Only tensors are supported as return types"
        )

        nresults = len(func_results)
        new_args = [memref_t(ty) for ty in func_results + func_inputs]

        @func_cif(*new_args, name=func_name)
        def f(*args):
            outputs = args[:nresults]
            inputs = args[nresults:]
            # convert input memrefs to tensors
            input_tensors = []
            for input in inputs:
                t = emit_buf_to_tensor(input, restrict=True)
                input_tensors.append(t)

            # keep track of cloned ops to replace operands
            cloned_map = {}

            def replace_operands(op: ir.Operation) -> ir.WalkResult:
                for i, oprnd in enumerate(op.operands):
                    if isinstance(oprnd, ir.BlockArgument):
                        # operand is block argument
                        owner = oprnd.owner.owner  # op that owns the block
                        if owner == target:
                            # payload func argument, replace with input tensor
                            op.operands[i] = input_tensors[oprnd.arg_number]
                    else:
                        # replace operands with cloned values
                        if oprnd.owner in cloned_map:
                            iresult = oprnd.result_number
                            op.operands[i] = cloned_map[oprnd.owner].results[iresult]
                return ir.WalkResult.ADVANCE

            for op in target.regions[0].blocks[0].operations:
                if isinstance(op, func.ReturnOp):
                    # emit materialize_in_destination for each return value
                    for i, res_val in enumerate(op.operands):
                        if res_val.owner not in cloned_map:
                            raise NotImplementedError("Unsupported return value")
                        iresult = res_val.result_number
                        new_val = cloned_map[res_val.owner].results[iresult]
                        bufferization.materialize_in_destination(
                            None,
                            new_val,
                            outputs[i],
                            restrict=True,
                            writable=True,
                        )
                else:
                    new_op = op.clone()
                    new_op.walk(replace_operands, ir.WalkOrder.PRE_ORDER)
                    cloned_map[op] = new_op

        return f.func_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ConvertFuncResultsToArgsOp",
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
                    new_func = ConvertFuncResultsToArgsOp.convert_func(target)
                    target.erase()
                converted_funcs.append(new_func)

            results.set_ops(op.converted_func, converted_funcs)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ConvertFuncResultsToArgsOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ConvertFuncResultsToArgsOp", effects):
            transform.consumes_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def convert_func_results_to_args(
    target: ir.Value[transform.AnyOpType],
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a ConvertFuncResultsToArgsOp."""
    op = ConvertFuncResultsToArgsOp(target=target)
    return op.converted_func

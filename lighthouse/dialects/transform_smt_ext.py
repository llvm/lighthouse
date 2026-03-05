from typing import overload, Sequence, Callable

from mlir import ir
from mlir.dialects import ext, smt, transform

from lighthouse.tune import trace

__all__ = [
    "ConstrainParamsOp",
    "TransformSMTDialectExtension",
    "constrain_params",
    "register_and_load",
]

def register_and_load(context=None):
    TransformSMTDialectExtension.load()


class TransformSMTDialectExtension(ext.Dialect, name="transform_smt_ext"):
    @classmethod
    def load(cls, *args, **kwargs):
        super(TransformSMTDialectExtension, cls).load(*args, **kwargs)

        for op in cls.operations:
            if hasattr(op, "attach_interfaces"):
                op.attach_interfaces()


class ConstrainParamsOp(
    TransformSMTDialectExtension.Operation, name="constrain_params"
):
    results_: Sequence[ext.Result[transform.AnyParamType]]
    params: Sequence[ext.Operand[transform.AnyParamType]]
    body_: ext.Region

    @property
    def body(self):
        return self.body_.blocks[0]

    @classmethod
    def attach_interfaces(cls, ctx=None):
        if not hasattr(cls, "_interfaces_attached"):
            cls.ConstrainParamsTransformOpInterfaceModel.attach(
                cls.OPERATION_NAME, context=ctx
            )
            cls.ConstrainParamsMemoryEffectsOpInterfaceModel.attach(
                cls.OPERATION_NAME, context=ctx
            )
            setattr(cls, "_interfaces_attached", True)

    class ConstrainParamsTransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ConstrainParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> transform.DiagnosedSilenceableFailure:
            env = dict()
            for operand in op.params:
                params = state.get_params(operand)
                assert len(params) == 1 and isinstance(params[0].value, int)
                env[operand] = trace.Constant(params[0].value)

            env = trace.trace_tune_and_smt_ops(op.operation, env)

            if not env[op].evaluate(env):  # evaluate the conjoined predicate
                return transform.DiagnosedSilenceableFailure.DefiniteFailure

            for result in op.results:
                res_value = env[result].evaluate(env)
                i64 = ir.IntegerType.get_signless(64)
                results.set_params(result, [ir.IntegerAttr.get(i64, res_value)])

            return transform.DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ConstrainParamsOp") -> bool:
            return False

    class ConstrainParamsMemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "ConstrainParamsOp", effects):
            if op.op_operands:
                transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.only_reads_payload(effects)


class MixedResultConstrainParamsOp(ConstrainParamsOp):
    def __init__(
        self,
        *args,
        result_values_or_types: Sequence[int | ir.Type],
        **kwargs,
    ):
        result_types = [
            res for res in result_values_or_types if isinstance(res, ir.Type)
        ]
        super().__init__(result_types, *args, **kwargs)
        op_results = iter(super().results)
        self._results = [
            next(op_results) if isinstance(res, ir.Type) else res
            for res in result_values_or_types
        ]

    @property
    def results(self) -> Sequence[int | ext.Result[transform.AnyParamType]]:
        return self._results


# class ConstrainParamsOpDecorator(ConstrainParamsOp):
#    def __init__(
#        self,
#        *params: transform.AnyParamType | int,
#        results: Sequence[int | ext.Result[transform.AnyParamType]] | None = None,
#        **kwargs,
#    ):
#        transform_params = [p for p in params if isinstance(p, ir.Value)]
#        super().__init__([], transform_params, **kwargs)
#        block_arg_types = [smt.IntType.get()] * len(transform_params)
#        self.body_.blocks.append(*block_arg_types)
#
#        self._arguments = []
#        self._results = results
#        smt_arguments = iter(self.body.arguments)
#        for param in params:
#            if isinstance(param, int):
#                self._arguments.append(param)
#            else:
#                self._arguments.append(next(smt_arguments))
#
#    @property
#    def results(self) -> Sequence[ext.Result | int]:
#        """Returns the yielded results of the decorated function, which are either
#        integers or the transform parameters that correspond to the yielded SMT
#        int values."""
#        assert self._results is not None, (
#            "Results are not available until the decorated function is called"
#        )
#        return self._results
#
#    def __call__(self, func):
#        with ir.InsertionPoint(self.body):
#            yielded_results = func(*self._arguments)
#
#            smt.yield_(res for res in yielded_results if isinstance(res, ir.Value))
#
#        print(f"{yielded_results=}")
#        if len(yielded_results) == 0:
#            return self
#
#        # In case of yielded results, we need to create a new ConstrainParamsOp with the same parameters and a body that contains the original body of the decorator, but with the yielded results as the results of the new op. We then replace the original op with the new one and return it.
#        result_types = [transform.AnyParamType.get()] * sum(
#            1 for res in yielded_results if isinstance(res, ir.Value)
#        )
#        with ir.InsertionPoint(self):
#            self_with_results = ConstrainParamsOp(
#                result_types, self.params, loc=self.location
#            )
#            self.body_.blocks[0].append_to(self_with_results.body_)
#            # new_block = self_with_results.body_.blocks.append(
#            #    *orig_block.arguments.types
#            # )
#            # arg_mapping = dict(zip(orig_block.arguments, new_block.arguments))
#            # lh_utils_rewrite.move_block(orig_block, new_block, arg_mapping)
#            # self.erase()
#
#        results = []
#        op_results = iter(self_with_results.results)
#        for yielded_result in yielded_results:
#            if isinstance(yielded_result, int):
#                results.append(yielded_result)
#            elif isinstance(yielded_result, ir.Value):
#                results.append(next(op_results))
#            else:
#                assert False, "Unsupported yielded result type"
#        setattr(self_with_results, "_results", results)
#        return self_with_results


@overload
def constrain_params(
    *params: ir.Value | int, loc=None, ip=None
) -> Callable[..., MixedResultConstrainParamsOp]: ...


@overload
def constrain_params(
    results: Sequence[ir.Type],
    params: Sequence[transform.AnyParamType],
    arg_types: Sequence[ir.Type],
    loc=None,
    ip=None,
) -> ConstrainParamsOp: ...


def constrain_params(
    *args, **kwargs
) -> ConstrainParamsOp | Callable[..., MixedResultConstrainParamsOp]:
    # The second overload:
    if len(args) == 0 or isinstance(args[0], ir.Type):
        arg_types = kwargs.pop("arg_types")
        op = ConstrainParamsOp(*args, **kwargs)
        op.body_.blocks.append(*arg_types)
        return op

    # The first overload:
    # return ConstrainParamsOpDecorator(*args, **kwargs)
    def wrapper(func):
        param_args = [p for p in args if isinstance(p, ir.Value)]
        constrain_params = ConstrainParamsOp([], param_args, **kwargs)
        constrain_params.body_.blocks.append(*[smt.IntType.get()] * len(param_args))

        block_args_iter = iter(constrain_params.body_.blocks[0].arguments)
        with ir.InsertionPoint(constrain_params.body):
            yielded_results = func(
                *(
                    next(block_args_iter) if isinstance(arg, ir.Value) else arg
                    for arg in args
                )
            )
            if not isinstance(yielded_results, Sequence):
                yielded_results = [yielded_results]
            smt.yield_(res for res in yielded_results if isinstance(res, ir.Value))

        if len(yielded_results) == 0:
            return constrain_params

        result_values_or_types = [
            transform.AnyParamType.get() if isinstance(res, ir.Value) else res
            for res in yielded_results
        ]

        mixed_result_op = MixedResultConstrainParamsOp(
            params=param_args, result_values_or_types=result_values_or_types, **kwargs
        )
        # Move the body of the original op to the version with (mixed) results.
        constrain_params.body_.blocks[0].append_to(mixed_result_op.body_)
        # Safe to remove as the op doesn't have results, so no users either.
        constrain_params.erase()
        return mixed_result_op

    return wrapper

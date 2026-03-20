from typing import overload, Sequence, Callable

from mlir import ir
from mlir.dialects import ext, smt, transform

from lighthouse.tune import trace

__all__ = [
    "ConstrainParamsOp",
    "TransformSMTExtensionDialect",
    "constrain_params",
    "register_and_load",
]


def register_and_load(**kwargs):
    """Register and load the TransformSMTDialectExtension and its operations."""

    TransformSMTExtensionDialect.load(**kwargs)


class TransformSMTExtensionDialect(ext.Dialect, name="transform_smt_ext"):
    """A Transform Dialect extension for SMT-related operations."""

    @classmethod
    def load(cls, *args, **kwargs):
        # Registers the dialect and its op classes and loads the dialect and ops into the context.
        super().load(*args, **kwargs)

        for op in cls.operations:
            if hasattr(op, "attach_interfaces"):
                op.attach_interfaces()


class ConstrainParamsOp(
    TransformSMTExtensionDialect.Operation, name="constrain_params"
):
    """Constrain transform params by SMT ops while also producing new params.

    In effect applies a predicate defined by the SMT ops in the body, which can
    reference the parameters as block arguments as !smt.int. The result params
    are defined by the !smt.int values yielded from the body.
    """

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
        """TransformOpInterface impl for evaluating the SMT constraints and producing new params."""

        @staticmethod
        def apply(
            op: "ConstrainParamsOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> transform.DiagnosedSilenceableFailure:
            # Set up the tracing environment by obtaining the transform params
            # and mapping them to Node constants, so that the traced Node
            # representation will refer to the params as just constants.
            env = dict()
            for operand in op.params:
                params = state.get_params(operand)
                assert len(params) == 1 and isinstance(params[0].value, int)
                env[operand] = trace.Constant(params[0].value)

            # Obtained traced representation of the body of the op.
            env = trace.trace_tune_and_smt_ops(op.operation, env)

            # Evaluate the predicate that represents the successful execution of the body.
            if not env[op].evaluate(env):
                return transform.DiagnosedSilenceableFailure.DefiniteFailure

            # If the predicate is satisfied, we can extract the values of the result params
            # from the environment and set them as the results of the transformation.
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
    """ConstrainParamsOp that supports both integer and SMTIntValues as results.

    Used to support `constrain_params` as a decorator on functions that yield a
    mix of Python integers and `!smt.int`s (which are either arguments to the
    function/block or the result of operations in the body). Upon the body's function
    returning, the original ConstrainParamsOp is replaced with this version
    that has the same parameters but whose `.results` corresponds to the mix of
    integers and SMT values yielded from the body.
    """

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


@overload
def constrain_params(
    *params: ir.Value | int, loc=None, ip=None
) -> Callable[..., MixedResultConstrainParamsOp]:
    """Calls the decorated function with param args converted to !smt.int args.

    The decorated function defines the body of the ConstrainParamsOp and handles
    args as `!smt.int` or Python integer. The function should yield a mix of
    Python integers and `!smt.int`s (the latter can be either block arguments or
    results of operations in the body). The original ConstrainParamsOp created
    by the decorator will be replaced with a MixedResultConstrainParamsOp that
    has the same parameters but whose results correspond to the mix of integers
    and SMT values yielded from the body.
    """

    ...


@overload
def constrain_params(
    results: Sequence[ir.Type],
    params: Sequence[transform.AnyParamType],
    loc=None,
    ip=None,
) -> ConstrainParamsOp:
    """Creates a ConstrainParamsOp where the body is defined by the caller."""

    ...


def constrain_params(
    *args, **kwargs
) -> ConstrainParamsOp | Callable[..., MixedResultConstrainParamsOp]:
    """Creates a ConstrainParamsOp or a decorator for a function that yields mixed results."""

    # The second overload:
    if len(args) == 0 or not (
        isinstance(args[0], ir.Value) or isinstance(args[0], int)
    ):
        params = kwargs.get("params") or args[1]
        arg_types = [smt.IntType.get()] * len(params)
        op = ConstrainParamsOp(*args, **kwargs)
        op.body_.blocks.append(*arg_types)
        return op

    # The first overload:
    def wrapper(func):
        # Create a ConstrainParamsOp with just the transform parameters as block arguments.
        param_args = [p for p in args if isinstance(p, ir.Value)]
        constrain_params = ConstrainParamsOp([], param_args, **kwargs)
        constrain_params.body_.blocks.append(*[smt.IntType.get()] * len(param_args))

        # Call `func` with !smt.int block arguments for corresponding transform params,
        # and just normal ints for those passed via `args`. The body of `func` will be
        # the body of the op, and it can yield a mix of Python integers and `!smt.int`s.
        # A corresponding `smt.yield` will be generated as the terminator.
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

        # In case no results are returned, the current ConstrainParamsOp is sufficient.
        if len(yielded_results) == 0:
            return constrain_params

        # Create a new version of the ConstrainParamsOp that has the same
        # parameters but whose results correspond to the mix of integers and
        # SMT values yielded from the body.
        result_values_or_types = [
            transform.AnyParamType.get() if isinstance(res, ir.Value) else res
            for res in yielded_results
        ]

        mixed_result_op = MixedResultConstrainParamsOp(
            params=param_args,
            result_values_or_types=result_values_or_types,
            **kwargs,
        )
        # Move the body of the original op to the version with (mixed) results.
        constrain_params.body_.blocks[0].append_to(mixed_result_op.body_)
        # Safe to remove as the op doesn't have results, so no users either.
        constrain_params.erase()
        return mixed_result_op

    return wrapper

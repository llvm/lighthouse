from typing import Sequence, Optional

from mlir import ir
from mlir.dialects import ext, transform, func, arith, scf, memref
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.utils.mlir import func_cif


def register_and_load(**kwargs):
    TransformExtensionDialect.load(**kwargs)


class TransformExtensionDialect(ext.Dialect, name="transform_ext"):
    @classmethod
    def load(cls, *args, **kwargs):
        # Registers the dialect and its op classes and loads the dialect and ops into the context.
        super().load(*args, **kwargs)

        # Attach interfaces to just registered/loaded operations.
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()


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
    param: ext.Result[transform.AnyParamType[()]] = ext.result(infer_type=True)

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


class ParamCmpEqOp(TransformExtensionDialect.Operation, name="param_cmp_eq"):
    """
    Compare the values of the `lhs` and `rhs` parameters for equality.

    The operation succeeds if the values are equal, and fails otherwise. If the
    parameters resolve to multiple values, the operation succeeds if all values
    are pairwise equal, and fails otherwise.
    """

    lhs: ext.Operand[transform.AnyParamType]
    rhs: ext.Operand[transform.AnyParamType]

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

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


def param_cmp_eq(lhs: ir.Value, rhs: ir.Value):
    return ParamCmpEqOp(lhs=lhs, rhs=rhs)


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


class WrapInBenchingFuncOp(
    TransformExtensionDialect.Operation, name="wrap_in_benching_func"
):
    """Create a function that calls `target` function in a benchmarking loop.

    The new function has the same arguments as `target` plus three additional ones:
    - A memref to store the timing results (one element per iteration).
    - The number of timed iterations.
    - The number of warmup iterations.
    """

    target: ext.Operand[transform.AnyOpType]
    bench_func: ext.Result[transform.AnyOpType[()]] = ext.result(infer_type=True)

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def wrap_in_benching_func(target: func.FuncOp, bench_name: str):
        """Create a function that calls `target` in a benchmarking loop.

        Each call to `target` is timed separately, and the times (in seconds)
        are stored in a memref that is passed as an additional argument to the
        benchmark function. It also takes two additional arguments for the
        number of runs and warmup iterations.
        """

        # define rtclock function
        f64_t = ir.F64Type.get()
        func.FuncOp("rtclock", ((), (f64_t,)), visibility="private")
        # emit benchmark function
        time_memref_t = ir.MemRefType.get((ir.ShapedType.get_dynamic_size(),), f64_t)
        index_t = ir.IndexType.get()
        args = target.type.inputs + [time_memref_t, index_t, index_t]

        @func_cif(*args, name=bench_name)
        def bench(*args):
            zero = arith.constant(index_t, 0)
            one = arith.constant(index_t, 1)
            func_args = list(args[: len(target.type.inputs)])
            times_memref, num_times, num_warmup = args[-3:]
            for i in scf.for_(zero, num_warmup, one):
                # FIXME(upstream): func.call needs to wrap _overridden_ CallOp.
                func.CallOp(target, func_args)
                scf.yield_(())
            # TODO: get `num_times` from the `times_memref`.
            for i in scf.for_(zero, num_times, one):
                tic = func.call((f64_t,), "rtclock", ())
                func.CallOp(target, func_args)
                toc = func.call((f64_t,), "rtclock", ())
                time = arith.subf(toc, tic)
                memref.store(time, times_memref, [i])
                scf.yield_(())

        return bench.func_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "WrapInBenchingFuncOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            if bench_name_attr := op.attributes.get("bench_name"):
                bench_name = bench_name_attr.value
                if len(targets) != 1:
                    return DiagnosedSilenceableFailure.SilenceableFailure
            else:
                bench_name = None

            bench_funcs = []
            for target in targets:
                if not isinstance(target, func.FuncOp):
                    return DiagnosedSilenceableFailure.SilenceableFailure

                with ir.InsertionPoint(target), target.location:
                    bench_func = WrapInBenchingFuncOp.wrap_in_benching_func(
                        target, bench_name or f"bench_{target.name.value}"
                    )
                    bench_funcs.append(bench_func)

            results.set_ops(op.bench_func, bench_funcs)

            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "WrapInBenchingFuncOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "WrapInBenchingFuncOp", effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def wrap_in_benching_func(
    target: ir.Value[transform.AnyOpType], bench_name: str | None = None
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create a WrapInBenchingFuncOp."""
    op = WrapInBenchingFuncOp(target=target)
    if bench_name is not None:
        op.attributes["bench_name"] = ir.StringAttr.get(bench_name)
    return op.bench_func

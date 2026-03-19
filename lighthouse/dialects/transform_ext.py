from mlir import ir
from mlir.dialects import ext, transform, func, arith, scf, memref
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.utils.mlir import func_cif


def register_and_load(context=None):
    TransformExtensionDialect.load()


class TransformExtensionDialect(ext.Dialect, name="transform_ext"):
    @classmethod
    def load(cls, *args, **kwargs):
        super().load(*args, **kwargs)
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()


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
    bench_func: ext.Result[transform.AnyOpType[()]]

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

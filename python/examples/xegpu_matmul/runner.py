import numpy as np
import ctypes
import os
from typing import Optional

from mlir.dialects import func, arith, scf, memref
from mlir.execution_engine import ExecutionEngine
from mlir import ir
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from lighthouse.utils import get_packed_arg
from mlir_utils import get_mlir_library_path


def get_engine(payload_module: ir.Module, opt_level: int = 3) -> ExecutionEngine:
    lib_dir = get_mlir_library_path()
    libs = [
        "libmlir_levelzero_runtime.so",
        "libmlir_runner_utils.so",
        "libmlir_c_runner_utils.so",
    ]
    libs = [os.path.join(lib_dir, lib) for lib in libs]
    execution_engine = ExecutionEngine(
        payload_module, opt_level=opt_level, shared_libs=libs
    )
    execution_engine.initialize()
    return execution_engine


def apply_transform_schedule(
    payload_module: ir.Module,
    schedule_module: ir.Module,
    dump_kernel: Optional[str] = None,
    dump_schedule: bool = False,
):
    if not dump_kernel or dump_kernel != "initial":
        # apply schedule on payload module
        named_seq = schedule_module.body.operations[0]
        named_seq.apply(payload_module)
    if dump_kernel:
        print(payload_module)
    if dump_schedule:
        print(schedule_module)


def lower_payload(
    workload: object,
    dump_kernel: Optional[str] = None,
    dump_schedule: bool = False,
    schedule_parameters: Optional[dict] = None,
) -> ir.Module:
    payload_module = workload.payload_module()
    schedule_module = workload.schedule_module(
        dump_kernel=dump_kernel, parameters=schedule_parameters
    )
    apply_transform_schedule(
        payload_module,
        schedule_module,
        dump_kernel=dump_kernel,
        dump_schedule=dump_schedule,
    )
    return payload_module


def execute(
    workload: object,
    check_correctness: bool = True,
    schedule_parameters: Optional[dict] = None,
    verbose: int = 0,
):
    # lower payload with schedule
    payload_module = lower_payload(workload, schedule_parameters=schedule_parameters)
    # get execution engine
    engine = get_engine(payload_module, requirements=workload.requirements())

    with workload.allocate(execution_engine=engine):
        # prepare function arguments
        inputs = workload.get_input_arrays(execution_engine=engine)
        pointers = [ctypes.pointer(ctypes.pointer(m)) for m in inputs]
        packed_args = get_packed_arg(pointers)

        # handle to payload function
        payload_func = engine.lookup(workload.payload_function_name)

        # call
        payload_func(packed_args)

        if check_correctness:
            workload.check_correctness(execution_engine=engine, verbose=verbose)


def benchmark(
    workload: object,
    nruns: int = 100,
    nwarmup: int = 10,
    schedule_parameters: Optional[dict] = None,
    check_correctness: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    # get original payload module
    payload_module = workload.payload_module()

    # find payload function
    payload_func = None
    for op in payload_module.operation.regions[0].blocks[0]:
        if (
            isinstance(op, func.FuncOp)
            and op.name.value == workload.payload_function_name
        ):
            payload_func = op
            break
    assert payload_func is not None, "Could not find payload function"
    payload_arguments = payload_func.type.inputs

    # emit benchmark function that calls payload and times it
    with ir.InsertionPoint(payload_module.body):
        # define rtclock function
        f64_t = ir.F64Type.get()
        func.FuncOp("rtclock", ((), (f64_t,)), visibility="private")
        # emit benchmark function
        time_memref_t = ir.MemRefType.get((nruns,), f64_t)
        args = payload_arguments + [time_memref_t]

        @func.func(*args)
        def benchmark(*args):
            index_t = ir.IndexType.get()
            zero = arith.constant(index_t, 0)
            one = arith.constant(index_t, 1)
            nwarmup_cst = arith.constant(index_t, nwarmup)
            for i in scf.for_(zero, nwarmup_cst, one):
                # FIXME(upstream): func.call is broken for this use case?
                func.CallOp(payload_func, list(args[: len(payload_arguments)]))
                scf.yield_(())
            nruns_cst = arith.constant(index_t, nruns)
            for i in scf.for_(zero, nruns_cst, one):
                tic = func.call((f64_t,), "rtclock", ())
                func.CallOp(payload_func, list(args[: len(payload_arguments)]))
                toc = func.call((f64_t,), "rtclock", ())
                time = arith.subf(toc, tic)
                memref.store(time, args[-1], [i])
                scf.yield_(())

        benchmark.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

    # lower
    apply_transform_schedule(
        payload_module,
        workload.schedule_module(parameters=schedule_parameters),
    )
    # get execution engine, rtclock requires mlir_c_runner
    engine = get_engine(payload_module)

    with workload.allocate(execution_engine=engine):
        inputs = workload.get_input_arrays(execution_engine=engine)
        pointers = [ctypes.pointer(ctypes.pointer(m)) for m in inputs]
        if check_correctness:
            # call payload once to verify correctness
            # prepare function arguments
            packed_args = get_packed_arg(pointers)

            payload_func = engine.lookup(workload.payload_function_name)
            payload_func(packed_args)
            success = workload.check_correctness(
                execution_engine=engine, verbose=verbose
            )
            if not success:
                raise ValueError("Benchmark verification failed.")

        # allocate buffer for timings and prepare arguments
        time_array = np.zeros((nruns,), dtype=np.float64)
        time_memref = get_ranked_memref_descriptor(time_array)
        time_pointer = ctypes.pointer(ctypes.pointer(time_memref))
        packed_args_with_time = get_packed_arg(pointers + [time_pointer])

        # call benchmark function
        benchmark_func = engine.lookup("benchmark")
        benchmark_func(packed_args_with_time)

    return time_array

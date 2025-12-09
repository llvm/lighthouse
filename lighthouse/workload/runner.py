"""
Utility functions for running workloads.
"""

import numpy as np
import os
from mlir import ir
from mlir.dialects import func, arith, scf, memref
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from lighthouse.utils.mlir import get_mlir_library_path
from lighthouse.utils.runtime.ffi import memrefs_to_packed_args
from lighthouse.workload import Workload
from typing import Optional


def get_engine(
    payload_module: ir.Module, shared_libs: list[str] = None, opt_level: int = 3
) -> ExecutionEngine:
    lib_dir = get_mlir_library_path()
    libs = []
    for so_file in shared_libs or []:
        so_path = os.path.join(lib_dir, so_file)
        if not os.path.isfile(so_path):
            raise ValueError(f"Could not find shared library {so_path}")
        libs.append(so_path)
    execution_engine = ExecutionEngine(
        payload_module, opt_level=opt_level, shared_libs=libs
    )
    execution_engine.initialize()
    return execution_engine


def execute(
    workload: Workload,
    check_correctness: bool = True,
    schedule_parameters: Optional[dict] = None,
    verbose: int = 0,
):
    # lower payload with schedule
    payload_module = workload.lower_payload(schedule_parameters=schedule_parameters)
    # get execution engine
    engine = get_engine(payload_module, shared_libs=workload.shared_libs())

    with workload.allocate_inputs(execution_engine=engine) as inputs:
        # prepare function arguments
        packed_args = memrefs_to_packed_args(inputs)

        # handle to payload function
        payload_func = engine.lookup(workload.payload_function_name)

        # call function
        payload_func(packed_args)

        if check_correctness:
            success = workload.check_correctness(
                execution_engine=engine, verbose=verbose
            )
            if not success:
                raise ValueError("Benchmark verification failed.")


def emit_benchmark_function(
    payload_module: ir.Module,
    payload_function_name: str,
    nruns: int,
    nwarmup: int,
):
    """
    Emit a benchmark function that calls payload function and times it.

    Every function call is timed separately. Returns the times (seconds) in a
    memref.
    """
    # find original payload function
    payload_func = None
    for op in payload_module.operation.regions[0].blocks[0]:
        if isinstance(op, func.FuncOp) and op.name.value == payload_function_name:
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


def benchmark(
    workload: Workload,
    nruns: int = 100,
    nwarmup: int = 10,
    schedule_parameters: Optional[dict] = None,
    check_correctness: bool = True,
    verbose: int = 0,
) -> np.ndarray:
    # get original payload module
    payload_module = workload.payload_module()

    # add benchmark function with timing
    emit_benchmark_function(
        payload_module, workload.payload_function_name, nruns, nwarmup
    )

    # lower
    schedule_module = workload.schedule_module(parameters=schedule_parameters)
    schedule_module.body.operations[0].apply(payload_module)

    # get execution engine, rtclock requires mlir_c_runner
    libs = workload.shared_libs()
    c_runner_lib = "libmlir_c_runner_utils.so"
    if c_runner_lib not in libs:
        libs.append(c_runner_lib)
    engine = get_engine(payload_module, shared_libs=libs)

    with workload.allocate_inputs(execution_engine=engine) as inputs:
        if check_correctness:
            # call payload once to verify correctness
            # prepare function arguments
            packed_args = memrefs_to_packed_args(inputs)

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
        packed_args_with_time = memrefs_to_packed_args(inputs + [time_memref])

        # call benchmark function
        benchmark_func = engine.lookup("benchmark")
        benchmark_func(packed_args_with_time)

    return time_array

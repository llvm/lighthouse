"""
Utility functions for running workloads.
"""

import numpy as np
import os
from mlir import ir
from mlir.dialects import func, arith, scf, memref
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from lighthouse.schedule.pattern_schedule import pattern_rewrite_schedule
from lighthouse.utils.mlir import func_cif, get_mlir_library_path
from lighthouse.utils.memref import to_packed_args
from lighthouse.workload import Workload
from typing import Optional


def get_engine(
    payload_module: ir.Module, shared_libs: list[str] = None, opt_level: int = 3
) -> ExecutionEngine:
    lib_dir = get_mlir_library_path()
    libs = []
    for so_file in shared_libs or []:
        # check if so_file is an absolute path
        so_path = so_file if os.path.isabs(so_file) else os.path.join(lib_dir, so_file)
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
    # get execution engine, rtclock requires mlir_c_runner
    libs = workload.shared_libs()
    c_runner_lib = "libmlir_c_runner_utils.so"
    if c_runner_lib not in libs:
        libs.append(c_runner_lib)
    engine = get_engine(payload_module, shared_libs=libs)

    with workload.allocate_inputs(execution_engine=engine) as inputs:
        # prepare function arguments
        packed_args = to_packed_args(inputs)

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


def bench_wrapper_pattern(funcname: str, benchname=None):
    """Returns a rewrite pattern that matches a function named `funcname` and clones it
    as a new function with name given by `benchname` (default: "bench_" + funcname).
    The new function is a benchmark wrapper that calls the payload function and times it.
    Every function call is timed separately. Returns the times (seconds) in a memref,
    which is passed as an additional argument to the benchmark function.
    It also takes two additional arguments for the number of runs and warmup iterations.
    """
    marker = "__bench_wrapped__"
    if benchname is None:
        benchname = f"bench_{funcname}"

    def match_and_rewrite(op, rewriter):
        if op.name.value != funcname:
            return True  # Failed match, return truthy value
        if marker in op.attributes:
            return True  # Already wrapped, skip
        payload_arguments = op.type.inputs

        with rewriter.ip, op.location:
            # define rtclock function
            f64_t = ir.F64Type.get()
            func.FuncOp("rtclock", ((), (f64_t,)), visibility="private")
            # emit benchmark function
            time_memref_t = ir.MemRefType.get(
                (ir.ShapedType.get_dynamic_size(),), f64_t
            )
            index_t = ir.IndexType.get()
            args = payload_arguments + [time_memref_t, index_t, index_t]

            @func_cif(*args, name=benchname)
            def bench(*args):
                index_t = ir.IndexType.get()
                zero = arith.constant(index_t, 0)
                one = arith.constant(index_t, 1)
                for i in scf.for_(zero, args[-1], one):
                    # FIXME(upstream): func.call is broken for this use case?
                    func.CallOp(op, list(args[: len(payload_arguments)]))
                    scf.yield_(())
                for i in scf.for_(zero, args[-2], one):
                    tic = func.call((f64_t,), "rtclock", ())
                    func.CallOp(op, list(args[: len(payload_arguments)]))
                    toc = func.call((f64_t,), "rtclock", ())
                    time = arith.subf(toc, tic)
                    memref.store(time, args[-3], [i])
                    scf.yield_(())

        # Mark original function as wrapped
        op.attributes[marker] = ir.UnitAttr.get()
        return None  # Success

    return match_and_rewrite


def get_bench_wrapper_schedule(workload: Workload):
    return pattern_rewrite_schedule(
        {
            "func.func": bench_wrapper_pattern(
                workload.payload_function_name,
                workload.benchmark_function_name,
            )
        },
        "add_bench_pattern",
    )


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

    # Lower payload with one or more schedules
    schedule_modules = workload.schedule_modules(parameters=schedule_parameters)
    for schedule_module in schedule_modules:
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
            packed_args = to_packed_args(inputs)

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
        packed_args_with_time = to_packed_args(inputs + [time_memref, nruns, nwarmup])

        # call benchmark function
        benchmark_func = engine.lookup(workload.benchmark_function_name)
        benchmark_func(packed_args_with_time)

    return time_array

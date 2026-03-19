"""
Utility functions for running workloads.
"""

import numpy as np
import os
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from lighthouse.dialects import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.utils.mlir import get_mlir_library_path
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


def get_bench_wrapper_schedule(workload: Workload):
    with schedule_boilerplate() as (schedule, named_seq):
        named_func = structured.structured_match(
            transform.AnyOpType.get(),
            target=named_seq.bodyTarget,
            ops={"func.func"},
            op_attrs={"sym_name": ir.StringAttr.get(workload.payload_function_name)},
        )
        bench_func = transform_ext.wrap_in_benching_func(
            named_func, bench_name=workload.benchmark_function_name
        )
        transform.yield_([bench_func])

    schedule.body.operations[0].verify()
    return schedule


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

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
from lighthouse.utils.memref import to_packed_args
from lighthouse.utils.mlir import get_mlir_library_path
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


def lower_payload(
    payload_module: ir.Module,
    schedule_modules: list[ir.Module],
    dump_payload: bool = False,
    dump_schedule: bool = False,
) -> ir.Module:
    """
    Apply transform schedules to the payload module.

    Optionally dumps the payload IR at the desired level and/or the
    transform schedules to stdout.

    Returns the lowered payload module.
    """
    if not isinstance(schedule_modules, list):
        raise TypeError(
            f"schedule_modules() must return a list of ir.Module instances, "
            f"got {type(schedule_modules).__name__}"
        )
    if not schedule_modules:
        raise ValueError("schedule_modules() must return at least one schedule module.")
    if not dump_payload or dump_payload != "initial":
        for schedule_module in schedule_modules:
            # apply schedule on payload module
            named_seq = schedule_module.body.operations[0]
            named_seq.apply(payload_module)
    if dump_payload:
        print(payload_module)
    if dump_schedule:
        for schedule_module in schedule_modules:
            print(schedule_module)
    return payload_module


def execute(
    workload: Workload,
    callback: Optional[callable] = None,
):
    # lower payload with schedule
    payload_module = lower_payload(
        workload.payload_module(),
        workload.schedule_modules(),
    )
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

        if callback is not None:
            callback(engine, inputs)


def get_bench_wrapper_schedule(workload: Workload):
    with schedule_boilerplate(result_types=[transform.any_op_t()]) as (
        schedule,
        named_seq,
    ):
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
    callback: Optional[callable] = None,
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
        # allocate buffer for timings and prepare arguments
        time_array = np.zeros((nruns,), dtype=np.float64)
        time_memref = get_ranked_memref_descriptor(time_array)
        packed_args_with_time = to_packed_args(inputs + [time_memref, nruns, nwarmup])

        # call benchmark function
        benchmark_func = engine.lookup(workload.benchmark_function_name)
        benchmark_func(packed_args_with_time)

        if callback is not None:
            callback(engine, inputs)

    return time_array

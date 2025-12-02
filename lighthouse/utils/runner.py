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
from lighthouse.utils import memrefs_to_packed_args
from lighthouse import Workload
from typing import Optional


def get_engine(payload_module, requirements=None, opt_level=3) -> ExecutionEngine:
    requirements = requirements or []
    context = ir.Context()
    location = ir.Location.unknown(context)
    required_libs = {
        "levelzero": (
            ["libmlir_levelzero_runtime.so"],
            "Did you compile LLVM with -DMLIR_ENABLE_LEVELZERO_RUNNER=1?",
        ),
        "mlir_runner": (["libmlir_runner_utils.so"], ""),
        "mlir_c_runner": (["libmlir_c_runner_utils.so"], ""),
    }
    libs = []
    lib_dir = os.path.join(get_mlir_library_path())
    for r in requirements:
        if r not in required_libs:
            raise ValueError(f"Unknown execution engine requirement: {r}")
        so_files, hint = required_libs[r]
        for f in so_files:
            so_path = os.path.join(lib_dir, f)
            if not os.path.isfile(so_path):
                msg = f"Could not find shared library {so_path}"
                if hint:
                    msg += "\n" + hint
                raise ValueError(msg)
            libs.append(so_path)
    with context, location:
        execution_engine = ExecutionEngine(
            payload_module, opt_level=opt_level, shared_libs=libs
        )
        execution_engine.initialize()
    return execution_engine


def lower_payload(
    workload,
    dump_kernel: Optional[str] = None,
    dump_schedule: bool = False,
    schedule_parameters: Optional[dict] = None,
) -> ir.Module:
    payload_module = workload.payload_module()
    schedule_module = workload.schedule_module(
        dump_kernel=dump_kernel, parameters=schedule_parameters
    )
    if not dump_kernel or dump_kernel != "initial":
        # apply schedule on payload module
        named_seq = schedule_module.body.operations[0]
        named_seq.apply(payload_module)
    if dump_kernel:
        print(payload_module)
    if dump_schedule:
        print(schedule_module)
    return payload_module


def execute(
    workload: Workload,
    check_correctness: bool = True,
    schedule_parameters: Optional[dict] = None,
    verbose: int = 0,
):
    # lower payload with schedule
    payload_module = lower_payload(workload, schedule_parameters=schedule_parameters)
    # get execution engine
    engine = get_engine(payload_module, requirements=workload.requirements())

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
    workload: Workload,
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
    emit_benchmark_function(payload_module, workload, nruns, nwarmup)

    # lower
    schedule_module = workload.schedule_module(parameters=schedule_parameters)
    schedule_module.body.operations[0].apply(payload_module)

    # get execution engine, rtclock requires mlir_c_runner
    requirements = workload.requirements()
    if "mlir_c_runner" not in requirements:
        requirements.append("mlir_c_runner")
    engine = get_engine(payload_module, requirements=requirements)

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

"""
Execution engine utility functions.
"""

import numpy as np
import ctypes
import os
from mlir import ir
from mlir.dialects.transform import interpreter as transform_interpreter
from mlir.dialects import func, arith, scf, memref
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from lighthouse.utils.mlir import get_mlir_library_path
from lighthouse.utils import get_packed_arg
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


def apply_transform_schedule(
    payload_module,
    schedule_module,
    context,
    location,
    dump_kernel: Optional[str] = None,
    dump_schedule: bool = False,
):
    if not dump_kernel or dump_kernel != "initial":
        with context, location:
            # invoke transform interpreter directly
            transform_interpreter.apply_named_sequence(
                payload_root=payload_module,
                transform_root=schedule_module.body.operations[0],
                transform_module=schedule_module,
            )
    if dump_kernel:
        print(payload_module)
    if dump_schedule:
        print(schedule_module)


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
    apply_transform_schedule(
        payload_module,
        schedule_module,
        workload.context,
        workload.location,
        dump_kernel=dump_kernel,
        dump_schedule=dump_schedule,
    )
    return payload_module


def execute(
    workload,
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

        # call function
        payload_func(packed_args)

        if check_correctness:
            success = workload.verify(execution_engine=engine, verbose=verbose)
            if not success:
                raise ValueError("Benchmark verification failed.")


def emit_benchmark_function(
    payload_module: ir.Module, workload: Workload, nruns: int, nwarmup: int
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
            and str(op.name).strip('"') == workload.payload_function_name
        ):
            payload_func = op
            break
    assert payload_func is not None, "Could not find payload function"
    payload_arguments = payload_func.type.inputs
    # emit benchmark function
    with workload.context, workload.location:
        with ir.InsertionPoint(payload_module.body):
            # define rtclock function
            f64_t = ir.F64Type.get()
            f = func.FuncOp("rtclock", ((), (f64_t,)), visibility="private")
            # emit new function
            time_memref_t = ir.MemRefType.get((nruns,), f64_t)
            args = payload_arguments + [time_memref_t]
            f = func.FuncOp("benchmark", (tuple(args), ()))
            f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        with ir.InsertionPoint(f.add_entry_block()):
            index_t = ir.IndexType.get()
            zero = arith.ConstantOp(index_t, 0)
            one = arith.ConstantOp(index_t, 1)
            # call payload for warmup runs
            nwarmup_cst = arith.ConstantOp(index_t, nwarmup)
            for_op = scf.ForOp(zero, nwarmup_cst, one)
            with ir.InsertionPoint(for_op.body):
                func.CallOp(payload_func, list(f.arguments[: len(payload_arguments)]))
                scf.YieldOp(())
            # call payload for benchmark runs, time every call separately
            nruns_cst = arith.ConstantOp(index_t, nruns)
            for_op = scf.ForOp(zero, nruns_cst, one)
            i = for_op.induction_variable
            with ir.InsertionPoint(for_op.body):
                tic = func.CallOp((f64_t,), "rtclock", ()).result
                func.CallOp(payload_func, list(f.arguments[: len(payload_arguments)]))
                toc = func.CallOp((f64_t,), "rtclock", ()).result
                time = arith.SubFOp(toc, tic)
                memref.StoreOp(time, f.arguments[-1], [i])
                scf.YieldOp(())
            func.ReturnOp(())


def benchmark(
    workload,
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
    apply_transform_schedule(
        payload_module,
        workload.schedule_module(parameters=schedule_parameters),
        workload.context,
        workload.location,
    )
    # get execution engine, rtclock requires mlir_c_runner
    requirements = workload.requirements()
    if "mlir_c_runner" not in requirements:
        requirements.append("mlir_c_runner")
    engine = get_engine(payload_module, requirements=requirements)

    with workload.allocate(execution_engine=engine):
        inputs = workload.get_input_arrays(execution_engine=engine)
        pointers = [ctypes.pointer(ctypes.pointer(m)) for m in inputs]
        if check_correctness:
            # call payload once to verify correctness
            # prepare function arguments
            packed_args = get_packed_arg(pointers)

            payload_func = engine.lookup(workload.payload_function_name)
            payload_func(packed_args)
            success = workload.verify(execution_engine=engine, verbose=verbose)
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

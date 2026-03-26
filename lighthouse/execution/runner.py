"""
Utility functions for running kernels.
"""

import numpy as np
import ctypes
import os
from contextlib import contextmanager
from functools import partial
from typing import Optional, Callable

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from lighthouse.dialects import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.utils.memref import to_packed_args
from lighthouse.utils.mlir import get_mlir_library_path
from .memory_manager import GPUMemoryManager, ShardMemoryManager, MemoryManager


@contextmanager
def numpy_to_memref_manager(inputs):
    yield [get_ranked_memref_descriptor(a) for a in inputs]


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
) -> ir.Module:
    """
    Apply transform schedules to the payload module.

    Returns the lowered payload module.
    """
    if not isinstance(schedule_modules, list):
        raise TypeError(
            f"schedule_modules() must return a list of ir.Module instances, "
            f"got {type(schedule_modules).__name__}"
        )
    if not schedule_modules:
        raise ValueError("schedule_modules() must return at least one schedule module.")
    for schedule_module in schedule_modules:
        # apply schedule on payload module
        named_seq = schedule_module.body.operations[0]
        named_seq.apply(payload_module)
    return payload_module


def get_bench_wrapper_schedule(payload_func: str, benchmark_func: str) -> ir.Module:
    with schedule_boilerplate(result_types=[transform.any_op_t()]) as (
        schedule,
        named_seq,
    ):
        named_func = structured.structured_match(
            transform.AnyOpType.get(),
            target=named_seq.bodyTarget,
            ops={"func.func"},
            op_attrs={"sym_name": ir.StringAttr.get(payload_func)},
        )
        bench_func = transform_ext.wrap_in_benching_func(
            named_func, bench_name=benchmark_func
        )
        transform.yield_([bench_func])

    schedule.body.operations[0].verify()
    return schedule


def _execute_kernel(
    payload_module: ir.Module,
    schedule_modules: list[ir.Module],
    mem_manager_cls: type | None = None,
    host_input_buffers: list[np.ndarray] | None = None,
    mem_manager_kwargs: dict = None,
    shared_libs: list[str] = None,
    payload_function_name: str = None,
    callback: Optional[
        Callable[[MemoryManager | None, list[ctypes.Structure]], None]
    ] = None,
    nruns: int = 100,
    nwarmup: int = 10,
    benchmark: bool = True,
) -> np.ndarray | None:
    if payload_function_name is None:
        raise ValueError("payload_function_name must be provided")

    # lower payload with schedule
    payload_module = lower_payload(payload_module, schedule_modules)

    # get execution engine, rtclock requires mlir_c_runner
    shared_libs = shared_libs or []
    c_runner_lib = "libmlir_c_runner_utils.so"
    if c_runner_lib not in shared_libs:
        shared_libs.append(c_runner_lib)
    engine = get_engine(payload_module, shared_libs=shared_libs)

    if mem_manager_cls in [None, GPUMemoryManager] and host_input_buffers is None:
        raise ValueError("host_input_buffers must be provided")

    if mem_manager_cls is None:
        mem_manager = None
        allocator = partial(numpy_to_memref_manager, host_input_buffers)
    elif mem_manager_cls is GPUMemoryManager:
        mem_manager = mem_manager_cls(engine)
        allocator = partial(mem_manager.clone_host_buffers, host_input_buffers)
    elif mem_manager_cls is ShardMemoryManager:
        if mem_manager_kwargs is None:
            raise ValueError("mem_manager_kwargs must be provided")
        mem_manager = mem_manager_cls(engine)
        allocator = partial(mem_manager.get_input_buffers, **mem_manager_kwargs)
    else:
        raise ValueError(f"Unsupported mem_manager_cls type: {mem_manager_cls}")

    if benchmark:
        time_array = np.zeros((nruns,), dtype=np.float64)
    else:
        time_array = None

    def _prepare_args(inputs):
        if benchmark:
            # allocate buffer for timings and prepare arguments
            time_memref = get_ranked_memref_descriptor(time_array)
            return to_packed_args(inputs + [time_memref, nruns, nwarmup])
        else:
            return to_packed_args(inputs)

    with allocator() as inputs:
        args = _prepare_args(inputs)

        # call function
        func = engine.lookup(payload_function_name)
        func(args)

        if callback is not None:
            callback(mem_manager, inputs)

    return time_array


def benchmark(
    payload_module: ir.Module,
    schedule_modules: list[ir.Module],
    host_input_buffers: list[np.ndarray] | None = None,
    mem_manager_cls: type | None = None,
    mem_manager_kwargs: dict = None,
    shared_libs: list[str] = None,
    payload_function_name: str = None,
    callback: Optional[
        Callable[[MemoryManager | None, list[ctypes.Structure]], None]
    ] = None,
    nruns: int = 100,
    nwarmup: int = 10,
) -> np.ndarray:
    return _execute_kernel(
        payload_module=payload_module,
        schedule_modules=schedule_modules,
        host_input_buffers=host_input_buffers,
        mem_manager_cls=mem_manager_cls,
        mem_manager_kwargs=mem_manager_kwargs,
        shared_libs=shared_libs,
        payload_function_name=payload_function_name,
        callback=callback,
        nruns=nruns,
        nwarmup=nwarmup,
        benchmark=True,
    )


def execute(
    payload_module: ir.Module,
    schedule_modules: list[ir.Module],
    host_input_buffers: list[np.ndarray] | None = None,
    mem_manager_cls: type | None = None,
    mem_manager_kwargs: dict = None,
    shared_libs: list[str] = None,
    payload_function_name: str = None,
    callback: Optional[
        Callable[[MemoryManager | None, list[ctypes.Structure]], None]
    ] = None,
) -> None:
    _execute_kernel(
        payload_module=payload_module,
        schedule_modules=schedule_modules,
        host_input_buffers=host_input_buffers,
        mem_manager_cls=mem_manager_cls,
        mem_manager_kwargs=mem_manager_kwargs,
        shared_libs=shared_libs,
        payload_function_name=payload_function_name,
        benchmark=False,
        callback=callback,
    )

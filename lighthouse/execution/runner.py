"""
Utility functions for running kernels.
"""

import typing
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
from .memory_manager import GPUMemoryManager, MemoryManager


class RunnerCallable(typing.Protocol):
    """
    Protocol for the argument access callback function used in the runner.
    """

    def __call__(
        self,
        inputs: list[ctypes.Structure],
        execution_engine: ExecutionEngine,
        memory_manager: Optional[MemoryManager],
    ) -> None: ...


class Runner:
    """
    Runner class for executing and benchmarking MLIR modules.
    """

    payload_benchmark_function_name: str = "__benchmark"

    def __init__(
        self,
        module: ir.Module,
        mem_manager_cls: type = None,
        shared_libs: list[str] = None,
        opt_level: int = 3,
    ):
        self.payload = module
        self.mem_manager_cls = mem_manager_cls
        if shared_libs is None:
            shared_libs = []
        # get execution engine, rtclock requires mlir_c_runner
        c_runner_lib = "libmlir_c_runner_utils.so"
        if c_runner_lib not in shared_libs:
            shared_libs.append(c_runner_lib)
        self.lib_dir = get_mlir_library_path()
        self.shared_libs = self._find_shared_libs(shared_libs)
        self.opt_level = opt_level
        self.engine = self._get_engine()

    def _find_shared_libs(self, shared_libs: list[str]) -> list[str]:
        """
        Find the shared libraries in the given directory that match the names in self.shared_libs.
        """
        found_libs = []
        for so_file in shared_libs:
            # check if so_file is an absolute path
            so_path = (
                so_file
                if os.path.isabs(so_file)
                else os.path.join(self.lib_dir, so_file)
            )
            if not os.path.isfile(so_path):
                raise ValueError(f"Could not find shared library {so_path}")
            found_libs.append(so_path)
        return found_libs

    def _get_engine(self) -> ExecutionEngine:
        """
        Get an execution engine for the given payload module, loading the necessary shared libraries.
        """
        execution_engine = ExecutionEngine(
            self.payload, opt_level=self.opt_level, shared_libs=self.shared_libs
        )
        execution_engine.initialize()
        return execution_engine

    @contextmanager
    def _numpy_to_memref_manager(self, inputs):
        """
        Context manager that yields memref descriptors for the given numpy input buffers.
        """
        yield [get_ranked_memref_descriptor(a) for a in inputs]

    def _execute_kernel(
        self,
        host_input_buffers: list[np.ndarray],
        payload_function_name: str = "",
        argument_access_callback: Optional[
            Callable[[list[ctypes.Structure], ExecutionEngine, MemoryManager], None]
        ] = None,
        nruns: int = 100,
        nwarmup: int = 10,
        benchmark: bool = True,
    ) -> np.ndarray | None:
        """
        Execute the payload module with the given pipeline and input buffers.

        If `mem_manager_cls` is provided, it will be used to allocate device buffers
        and copy data from host input buffers.

        The `argument_access_callback` function can be used to (read/write) access the buffers
        after execution. The callback signature is

        `argument_access_callback(inputs, execution_engine=..., memory_manager=...)`

        where `inputs` is the list of memref descriptors for the input buffers,
        `execution_engine` is the execution engine instance, and `memory_manager`
        is the memory manager instance (or `None` if no memory manager is used).
        """

        if host_input_buffers is None:
            raise ValueError("host_input_buffers must be provided")

        if self.mem_manager_cls is None:
            mem_manager = None
            allocator = partial(self._numpy_to_memref_manager, host_input_buffers)
        elif self.mem_manager_cls is GPUMemoryManager:
            mem_manager = self.mem_manager_cls(self.engine)
            allocator = partial(mem_manager.clone_host_buffers, host_input_buffers)
        else:
            raise ValueError(
                f"Unsupported mem_manager_cls type: {self.mem_manager_cls}"
            )

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
            if benchmark:
                function_name = self.payload_benchmark_function_name
            else:
                function_name = payload_function_name
            func = self.engine.lookup(function_name)
            func(args)

            if argument_access_callback is not None:
                argument_access_callback(
                    inputs, execution_engine=self.engine, memory_manager=mem_manager
                )

        return time_array

    def benchmark(
        self,
        host_input_buffers: list[np.ndarray],
        argument_access_callback: RunnerCallable = None,
        nruns: int = 100,
        nwarmup: int = 10,
    ) -> np.ndarray:
        """
        Benchmark the payload module with the given pipeline and input buffers.
        """
        return self._execute_kernel(
            host_input_buffers=host_input_buffers,
            argument_access_callback=argument_access_callback,
            nruns=nruns,
            nwarmup=nwarmup,
            benchmark=True,
        )

    def execute(
        self,
        payload_function_name: str,
        host_input_buffers: list[np.ndarray],
        argument_access_callback: RunnerCallable = None,
    ) -> None:
        """
        Execute the payload module with the given pipeline and input buffers.
        """
        self._execute_kernel(
            host_input_buffers=host_input_buffers,
            payload_function_name=payload_function_name,
            argument_access_callback=argument_access_callback,
            benchmark=False,
        )

    @staticmethod
    def get_bench_wrapper_schedule(payload_func: str) -> ir.Module:
        """
        Get a schedule that wraps the payload function in a benchmarking function.
        The function name is defined in Runner and will be used by the runner benchmark method.
        """
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
                named_func, bench_name=Runner.payload_benchmark_function_name
            )
            transform.yield_([bench_func])

        schedule.body.operations[0].verify()
        return schedule

    @staticmethod
    def get_gpu_argument_access_callback(
        output_shape: tuple[int, int], dtype: np.dtype
    ) -> tuple[np.ndarray, RunnerCallable]:
        D_host_copy = np.zeros(output_shape, dtype=dtype)

        def argument_access_callback(
            inputs: list[ctypes.Structure],
            *,
            memory_manager: GPUMemoryManager,
            **kwargs,
        ):
            memory_manager.copy(inputs[0], D_host_copy)

        return D_host_copy, argument_access_callback

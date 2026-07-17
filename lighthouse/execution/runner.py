"""
Infrastructure for running kernels.
"""

import typing
import numpy as np
import ctypes
import os
from contextlib import contextmanager
from functools import partial
from collections.abc import Callable

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from lighthouse.execution.target import TargetInfo
from lighthouse.utils.sys_config import enable_amx
from lighthouse.dialects.transform import transform_ext
from lighthouse.schedule import schedule_boilerplate
from lighthouse.utils.memref import to_packed_args
from lighthouse.utils.mlir import get_mlir_library_path
from lighthouse.utils.lib_finder import find_openmp_library
from .memory_manager import GPUMemoryManager, ExternalMemoryManager, MemoryManager


class RunnerCallable(typing.Protocol):
    """
    Protocol for the argument access callback function used in the runner.
    """

    def __call__(
        self,
        inputs: list[ctypes.Structure],
        execution_engine: ExecutionEngine,
        memory_manager: MemoryManager | None,
    ) -> None: ...


class Runner:
    """
    Runner class for executing and benchmarking MLIR modules.
    """

    payload_benchmark_function_name: str = "__benchmark"

    def __init__(
        self,
        module: ir.Module,
        mem_manager_cls: type | None = None,
        shared_libs: list[str] | None = None,
        opt_level: int = 3,
        target: TargetInfo = None,
    ):
        self.payload = module
        self.target = target if target else TargetInfo()
        self.mem_manager_cls = mem_manager_cls
        if shared_libs is None:
            shared_libs = []
        # get execution engine, rtclock requires mlir_c_runner
        c_runner_lib = "libmlir_c_runner_utils.so"
        if c_runner_lib not in shared_libs:
            shared_libs.append(c_runner_lib)
        self.lib_dir = get_mlir_library_path()
        shared_libs = self._find_shared_libs(shared_libs)
        # Remove duplicates, the same library cannot be loaded multiple times.
        self.shared_libs = list(dict.fromkeys(shared_libs))
        self.opt_level = opt_level
        self.engine = self._get_engine()
        self._configure_system()

    def _configure_system(self):
        """
        Apply target-specific configurations.
        """
        # Enable AMX register support.
        if self.target.is_supported("amx"):
            enable_amx()

    def _uses_openmp(self) -> bool:
        """
        Return True if the payload contains OpenMP dialect operations.

        Their presence means the jitted code must be linked against an OpenMP library.
        """
        found = False

        def match(op: ir.Operation) -> ir.WalkResult:
            nonlocal found
            if op.name.startswith("omp."):
                found = True
                return ir.WalkResult.INTERRUPT
            return ir.WalkResult.ADVANCE

        self.payload.operation.walk(match)
        return found

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

        # The OpenMP runtime lives outside the MLIR library directory and must
        # be located on the host when the payload was parallelized with OpenMP.
        if self._uses_openmp():
            omp_lib = find_openmp_library()
            if omp_lib is None:
                raise RuntimeError(
                    "Payload requires the OpenMP runtime but no OpenMP library "
                    "could be found on the system."
                )
            found_libs.append(omp_lib)

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
        host_input_buffers: list,
        payload_function_name: str = "",
        argument_access_callback: Callable[
            [list[ctypes.Structure], ExecutionEngine, MemoryManager], None
        ]
        | None = None,
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
            if any(not isinstance(buf, np.ndarray) for buf in host_input_buffers):
                raise ValueError(
                    "host_input_buffers must be numpy arrays when no mem_manager_cls is provided"
                )
            mem_manager = None
            allocator = partial(self._numpy_to_memref_manager, host_input_buffers)
        elif self.mem_manager_cls is GPUMemoryManager:
            mem_manager = self.mem_manager_cls(self.engine)
            allocator = partial(mem_manager.clone_host_buffers, host_input_buffers)
        elif issubclass(self.mem_manager_cls, ExternalMemoryManager):
            mem_manager = self.mem_manager_cls(self.engine)
            allocator = partial(mem_manager.get_memrefs, host_input_buffers)
        else:
            raise ValueError(
                f"Unsupported mem_manager_cls type: {self.mem_manager_cls}"
            )

        with allocator() as inputs:
            # call function
            if benchmark:
                # allocate buffer for timings and prepare arguments
                time_array = np.zeros((nruns,), dtype=np.float64)
                time_memref = get_ranked_memref_descriptor(time_array)
                args = to_packed_args(inputs + [time_memref, nruns, nwarmup])

                # Run the benchmark function instead of the main one
                function_name = self.payload_benchmark_function_name
            else:
                # No need for extra allocations
                time_array = None
                args = to_packed_args(inputs)

                # Run the main function
                function_name = payload_function_name

            # Now lookup and call the function
            func = self.engine.lookup(function_name)
            func(args)

            # If an argument access callback is provided,
            # use it to recover the output data from the device after execution.
            if argument_access_callback is not None:
                argument_access_callback(
                    inputs, execution_engine=self.engine, memory_manager=mem_manager
                )

        return time_array

    def benchmark(
        self,
        host_input_buffers: list,
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
        host_input_buffers: list,
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

    def dump_object_file(self, file_name: str) -> str:
        """
        Dump the compiled object file.

        Args:
            file_name: Target output file.

        Returns:
            Name of the dumped file.
        """
        if not file_name:
            raise ValueError("non-empty file_name must be provided")
        self.engine.dump_to_object_file(file_name)
        return file_name

    @staticmethod
    def get_bench_wrapper_schedule(payload_func: str) -> ir.Module:
        """
        Get a schedule that wraps the payload function in a benchmarking function.
        The function name is defined in Runner and will be used by the runner benchmark method.
        This schedule must apply to the module before any other in an optimizing pipeline.
        """
        with ir.Location.unknown():
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
        host_buffer: np.ndarray,
        arg_index: int = 0,
    ) -> RunnerCallable:
        """Returns a callback that copies device-allocated function argument to the host."""

        def argument_access_callback(
            inputs: list[ctypes.Structure],
            *,
            memory_manager: GPUMemoryManager,
            **kwargs,
        ):
            memory_manager.copy(inputs[arg_index], host_buffer)

        return argument_access_callback

    @staticmethod
    def make_function_callable(module: ir.Module, func_name: str) -> None:
        """
        Set the 'llvm.emit_c_interface' attribute of the given function in the module.
        This is required to make the function callable from the execution engine.
        It has to be called on a @func.func (not an @llvm.func), so should be called
        before the LLVM lowering stages are added to the pipeline.
        """
        with module.context:
            for func in module.body.operations:
                if func.sym_name.value == func_name:
                    func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                    break

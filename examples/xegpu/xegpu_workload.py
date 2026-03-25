from dataclasses import dataclass

from lighthouse.workload import Workload
import numpy as np
from contextlib import contextmanager
from abc import ABC, abstractmethod

from mlir.execution_engine import ExecutionEngine

from lighthouse.workload import MemoryManager, GPUMemoryManager


def matmul_complexity(
    M: int,
    N: int,
    K: int,
    bias: bool,
    relu: bool,
    accumulate_c: bool,
    nbytes_ab: int,
    nbytes_c: int,
):
    """Complexity of matmul operation with optional post-ops"""
    flop_count = 2 * M * N * K
    memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
    memory_writes = M * N * nbytes_c  # write C
    # Below we assume the post-ops are tiled-and-fused and do not cause
    # reads/writes to global memory.
    if bias:
        flop_count += M * N
        memory_reads += N * nbytes_c  # read bias vector
    if relu:
        flop_count += M * N
    if accumulate_c:
        memory_reads += M * N * nbytes_c  # read C for accumulation
    return flop_count, memory_reads, memory_writes


@dataclass
class XeGPUWorkload(Workload, ABC):
    """
    Base class for XeGPU workloads.

    Handles device buffer allocation/deallocation.
    """

    memory_manager_class: type[MemoryManager] = GPUMemoryManager
    memory_manager: MemoryManager | None = None

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        if self.memory_manager is None:
            self.memory_manager = self.memory_manager_class(execution_engine)
        host_arrays = [a for a in self._initial_host_arrays if a is not None]
        with self.memory_manager.clone_host_buffers(host_arrays) as device_buffers:
            yield device_buffers

    @abstractmethod
    def _initial_host_arrays(self) -> list[np.ndarray]:
        pass

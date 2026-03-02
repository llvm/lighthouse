from lighthouse.workload import Workload
import ctypes
from contextlib import contextmanager
from abc import ABC, abstractmethod

import numpy as np
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    as_ctype,
)
from mlir.execution_engine import ExecutionEngine

from lighthouse.utils.memref import to_ctype as memref_to_ctype


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


class XeGPUWorkload(Workload, ABC):
    """
    Base class for XeGPU workloads.

    Handles device buffer allocation/deallocation.
    """

    def __init__(self):
        # cache allocated memrefs
        self.gpu_memrefs = {}

    def _allocate_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype_str: str,
        execution_engine: ExecutionEngine,
    ) -> ctypes.Structure:
        key = (name, dtype_str)
        if key in self.gpu_memrefs:
            return self.gpu_memrefs[key]
        dtype = {
            "f16": np.float16,
            "f32": np.float32,
        }[dtype_str]
        mref = make_nd_memref_descriptor(len(shape), as_ctype(dtype))()
        ptr_mref = memref_to_ctype(mref)
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        execution_engine.invoke("gpu_alloc_" + dtype_str, ptr_mref, *ptr_dims)
        self.gpu_memrefs[key] = mref
        return mref

    def _deallocate_all(self, execution_engine: ExecutionEngine):
        for (_, dtype_str), mref in self.gpu_memrefs.items():
            ptr_mref = ctypes.pointer(ctypes.pointer(mref))
            execution_engine.invoke("gpu_dealloc_" + dtype_str, ptr_mref)
        self.gpu_memrefs = {}

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            yield self._get_input_arrays(execution_engine)
        finally:
            self._deallocate_all(execution_engine)

    @abstractmethod
    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        pass

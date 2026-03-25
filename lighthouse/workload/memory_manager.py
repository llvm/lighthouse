import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, field
import abc
import numpy as np

from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_mlir_type, mlir_to_numpy_dtype
from lighthouse.utils.numpy import numpy_to_ctype
from lighthouse.ingress.mlir_gen.shard_utils import (
    emit_dealloc,
    emit_shard_alloc,
    emit_shard_gather,
)

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    as_ctype,
)


@dataclass
class MemoryManager(abc.ABC):
    """Abstract base class for memory management."""

    execution_engine: ExecutionEngine

    @abc.abstractmethod
    def alloc(self, name: str = None, **kwargs) -> ctypes.Structure:
        """Allocate a device buffer and return a memref descriptor."""
        pass

    @abc.abstractmethod
    def get(self, name: str) -> ctypes.Structure:
        """Look up a previously allocated device buffer by name."""
        pass

    @abc.abstractmethod
    def deallocate_all(self):
        """Deallocate all previously allocated device buffers."""
        pass

    @staticmethod
    def emit_memory_management_funcs(payload_module: ir.Module, **kwargs):
        """Emit utility functions required by this memory manager into the payload module."""
        pass


@dataclass
class DeviceMemoryManager(MemoryManager, abc.ABC):
    """Abstract base class for handling memory on accelerator devices."""

    @abc.abstractmethod
    def copy(
        self, src: ctypes.Structure | np.ndarray, dst: ctypes.Structure | np.ndarray
    ):
        """Copy data to/from externally allocated buffers."""
        pass

    @abc.abstractmethod
    def clone_host_buffers(
        self, host_inputs: list[np.ndarray], names: list[str] = None
    ):
        """Context manager for creating device buffers from host inputs."""
        pass


@dataclass
class GPUMemoryManager(DeviceMemoryManager):
    """GPU memory manager that uses MLIR gpu.alloc/dealloc/memcpy ops."""

    allocated_buffers: dict[str, tuple[ctypes.Structure, type]] = field(
        default_factory=dict
    )
    buf_counter: int = 0

    def alloc(
        self, shape: tuple[int, ...], elem_type: type, name: str = None
    ) -> ctypes.Structure:
        if name is None:
            name = f"buffer_{self.buf_counter}"
            self.buf_counter += 1
        assert name not in self.allocated_buffers, (
            f"Buffer with name '{name}' already exists."
        )
        np_dtype = mlir_to_numpy_dtype(elem_type)
        mref = make_nd_memref_descriptor(len(shape), as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        rank = len(shape)
        assert rank in (1, 2), "Only 1d or 2d arrays are supported."
        suffix = f"{rank}d_{str(elem_type)}"
        self.execution_engine.invoke("gpu_alloc_" + suffix, ptr_mref, *ptr_dims)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers[name] = mref, elem_type
        return mref

    def get(self, name: str) -> ctypes.Structure:
        assert name in self.allocated_buffers, f"No buffer found with name '{name}'."
        return self.allocated_buffers[name][0]

    def deallocate_all(self):
        for mref, elem_type in self.allocated_buffers.values():
            ptr_mref = memref_to_ctype(mref)
            rank = len(mref.shape)
            suffix = f"{rank}d_{elem_type}"
            self.execution_engine.invoke("gpu_dealloc_" + suffix, ptr_mref)
        self.allocated_buffers.clear()
        self.buf_counter = 0

    def copy(
        self, src: ctypes.Structure | np.ndarray, dst: ctypes.Structure | np.ndarray
    ):
        """Copy data between host and device buffers. Host buffer must be a numpy array."""
        if not isinstance(src, np.ndarray) and not isinstance(dst, np.ndarray):
            raise ValueError("At least one of src or dst must be a numpy array.")
        rank = type_str = None
        if isinstance(src, np.ndarray):
            rank = len(src.shape)
            type_str = str(numpy_to_mlir_type(src.dtype))
            src_ctype = numpy_to_ctype(src)
        else:
            src_ctype = memref_to_ctype(src)
        if isinstance(dst, np.ndarray):
            rank = len(dst.shape)
            type_str = str(numpy_to_mlir_type(dst.dtype))
            dst_ctype = numpy_to_ctype(dst)
        else:
            dst_ctype = memref_to_ctype(dst)
        copy_func_name = f"gpu_copy_{rank}d_{type_str}"
        self.execution_engine.invoke(copy_func_name, src_ctype, dst_ctype)

    @contextmanager
    def clone_host_buffers(
        self, host_inputs: list[np.ndarray], names: list[str] = None
    ):
        buffers = []
        try:
            for i, host_arr in enumerate(host_inputs):
                name = names[i] if names is not None else None
                buf = self.alloc(
                    host_arr.shape, numpy_to_mlir_type(host_arr.dtype), name=name
                )
                self.copy(host_arr, buf)
                buffers.append(buf)
            yield buffers
        finally:
            self.deallocate_all()

    @staticmethod
    def emit_memory_management_funcs(
        payload_module: ir.Module,
        host_inputs: list[np.ndarray] = None,
        ranks_and_types: list[tuple[int, type]] = None,
    ):
        """Emit utility functions required by this class into the payload module."""
        assert host_inputs is not None or ranks_and_types is not None, (
            "Either host_inputs or ranks_and_types must be provided"
        )
        if host_inputs is not None:
            ranks_and_types = set(
                (arr.ndim, numpy_to_mlir_type(arr.dtype)) for arr in host_inputs
            )
        else:
            ranks_and_types = set(ranks_and_types)
        with ir.InsertionPoint(payload_module.body):
            for rank, elem_type in ranks_and_types:
                emit_gpu_util_funcs(elem_type, rank)


@dataclass
class ShardMemoryManager(MemoryManager):
    """Memory manager that handles memory allocations for shard dialect."""

    allocated_buffers: dict[str, tuple[ctypes.Structure, type, str, int]] = field(
        default_factory=dict
    )
    buf_counter: int = 0

    def alloc(
        self, elem_type: type, name: str = None, kind: str = None, rank: int = None
    ) -> ctypes.Structure:
        if kind is None:
            raise ValueError("kind must be provided")
        if rank is None:
            raise ValueError("rank must be provided")
        if name is None:
            name = f"buffer_{self.buf_counter}"
            self.buf_counter += 1
        assert name not in self.allocated_buffers, f"Buffer '{name}' already exists"

        np_dtype = mlir_to_numpy_dtype(elem_type)
        mref = make_nd_memref_descriptor(rank, as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        self.execution_engine.invoke("alloc_" + kind, ptr_mref)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers[name] = (mref, elem_type, kind, rank)
        return mref

    def gather(self, name: str, elem_type: type) -> ctypes.Structure:
        assert name in self.allocated_buffers, f"No buffer found with name '{name}'."
        _, stored_elem_type, kind, rank = self.allocated_buffers[name]
        assert stored_elem_type == elem_type, (
            f"Element type mismatch: {stored_elem_type} != {elem_type}."
        )
        assert "gathered_" + name not in self.allocated_buffers, (
            f"Buffer '{name}' already gathered"
        )

        np_dtype = mlir_to_numpy_dtype(elem_type)
        mref = make_nd_memref_descriptor(rank, as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        src = memref_to_ctype(self.get(name))
        self.execution_engine.invoke("gather_" + kind, ptr_mref, src)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers["gathered_" + name] = (mref, elem_type, kind, rank)
        return mref

    def get(self, name: str) -> ctypes.Structure:
        assert name in self.allocated_buffers, f"No buffer found with name '{name}'."
        return self.allocated_buffers[name][0]

    def deallocate_all(self):
        for mref, _, _, rank in self.allocated_buffers.values():
            self.execution_engine.invoke(f"dealloc_{rank}d", memref_to_ctype(mref))
        self.allocated_buffers.clear()
        self.buf_counter = 0

    @contextmanager
    def get_input_buffers(
        self,
        kinds_and_ranks: list[tuple[str, int]],
        elem_type: type,
        names: list[str] = None,
        init_func: callable = None,
    ):
        """Context manager for looking up previously allocated buffers by name."""
        if names is None:
            names = [None] * len(kinds_and_ranks)
        buffers = []
        try:
            for (kind, rank), name in zip(kinds_and_ranks, names):
                buf = self.alloc(elem_type, kind=kind, name=name, rank=rank)
                if init_func is not None:
                    init_func(buf)
                buffers.append(buf)
            yield buffers
        finally:
            self.deallocate_all()

    @staticmethod
    def emit_memory_management_funcs(
        payload_module: ir.Module,
        shapes: list[tuple[str, list, list]],
        elem_type: type,
        grid: str = "grid0",
    ):
        """Emit utility functions required by this class into the payload module."""

        ranks = set(len(shape) for _, shape, _ in shapes)
        with ir.InsertionPoint(payload_module.body):
            for name, shape, split in shapes:
                rank = len(shape)
                tensor_type = ir.RankedTensorType.get(shape, elem_type)
                emit_shard_alloc(name, grid, tensor_type, split)
                emit_shard_gather(name, grid, tensor_type, split)
            for rank in ranks:
                emit_dealloc(elem_type, rank)

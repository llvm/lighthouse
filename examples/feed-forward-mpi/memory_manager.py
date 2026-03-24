import ctypes
from contextlib import contextmanager
from dataclasses import dataclass, field

from mlir import ir
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    as_ctype,
)

from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.workload.memory_manager import MemoryManager
from ff_weight_stationary import _emit_alloc, _emit_gather, _emit_dealloc_2d


@dataclass
class ShardMemoryManager(MemoryManager):
    """Memory manager that handles memory allocations for shard dialect."""

    allocated_buffers: dict[str, ctypes.Structure] = field(default_factory=dict)
    buffer_kinds: dict[str] = field(default_factory=dict)
    buffer_elem_types: dict[str, type] = field(default_factory=dict)
    buf_counter: int = 0

    def alloc(
        self, elem_type: type, kind: str = None, name: str = None
    ) -> ctypes.Structure:
        if kind is None:
            raise ValueError(
                "kind must be provided for ShardMemoryManager allocations."
            )
        if name is None:
            name = f"buffer_{self.buf_counter}"
            self.buf_counter += 1
        assert name not in self.allocated_buffers, f"Buffer '{name}' already exists"

        np_dtype = mlir_to_numpy_dtype(elem_type)
        mref = make_nd_memref_descriptor(2, as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        self.execution_engine.invoke("alloc_" + kind, ptr_mref)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers[name] = mref
        self.buffer_elem_types[name] = elem_type
        self.buffer_kinds[name] = kind
        return mref

    def gather(self, name: str, elem_type: type) -> ctypes.Structure:
        assert name in self.allocated_buffers, f"No buffer found with name '{name}'."
        assert self.buffer_elem_types[name] == elem_type, (
            f"Element type mismatch: {self.buffer_elem_types[name]} != {elem_type}."
        )
        assert "gathered_" + name not in self.allocated_buffers, (
            f"Buffer '{name}' already gathered"
        )

        np_dtype = mlir_to_numpy_dtype(elem_type)
        mref = make_nd_memref_descriptor(2, as_ctype(np_dtype))()
        ptr_mref = memref_to_ctype(mref)
        src = memref_to_ctype(self.get(name))
        kind = self.buffer_kinds[name]
        self.execution_engine.invoke("gather_" + kind, ptr_mref, src)

        # NOTE need to track datatype as MemRefDescriptor does not include element type
        self.allocated_buffers["gathered_" + name] = mref
        self.buffer_elem_types["gathered_" + name] = elem_type
        self.buffer_kinds["gathered_" + name] = kind
        return mref

    def get(self, name: str) -> ctypes.Structure:
        assert name in self.allocated_buffers, f"No buffer found with name '{name}'."
        return self.allocated_buffers[name]

    def deallocate_all(self):
        for mref in self.allocated_buffers.values():
            self.execution_engine.invoke("dealloc_2d", memref_to_ctype(mref))
        self.allocated_buffers.clear()
        self.buffer_elem_types.clear()

    @contextmanager
    def get_input_buffers(
        self,
        kinds: list[str],
        elem_type: type,
        names: list[str] = None,
        init_func: callable = None,
    ):
        """Context manager for looking up previously allocated buffers by name."""
        if names is None:
            names = [None] * len(kinds)
        buffers = []
        try:
            for kind, name in zip(kinds, names):
                buf = self.alloc(elem_type, kind=kind, name=name)
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
    ):
        """Emit utility functions required by this class into the payload module."""

        with ir.InsertionPoint(payload_module.body):
            for name, shape, split in shapes:
                tensor_type = ir.RankedTensorType.get(shape, elem_type)
                _emit_alloc(name, tensor_type, split)
                _emit_gather(name, tensor_type, split)

            _emit_dealloc_2d(elem_type)

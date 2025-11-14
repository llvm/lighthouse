import ctypes

from itertools import accumulate
from functools import reduce
import operator

import mlir.runtime.np_to_memref as np_mem


class MemRefManager:
    """
    A utility class for manual management of MLIR memrefs.

    When used together with memref operation from within a jitted MLIR module,
    it is assumed that Memref dialect allocations and deallocation are performed
    through standard runtime `malloc` and `free` functions.

    Custom allocators are currently not supported. For more details, see:
    https://mlir.llvm.org/docs/TargetLLVMIR/#generic-alloction-and-deallocation-functions
    """

    def __init__(self) -> None:
        # Library name is left unspecified to allow for symbol search
        # in the global symbol table of the current process.
        # For more details, see:
        # https://github.com/python/cpython/issues/78773
        self.dll = ctypes.CDLL(name=None)
        self.fn_malloc = self.dll.malloc
        self.fn_malloc.argtypes = [ctypes.c_size_t]
        self.fn_malloc.restype = ctypes.c_void_p
        self.fn_free = self.dll.free
        self.fn_free.argtypes = [ctypes.c_void_p]
        self.fn_free.restype = None

    def alloc(self, *shape: int, ctype: ctypes._SimpleCData) -> ctypes.Structure:
        """
        Allocate an empty memory buffer.
        Returns an MLIR ranked memref descriptor.

        Args:
            shape: A sequence of integers defining the buffer's shape.
            ctype: A C type of buffer's elements.
        """
        assert issubclass(ctype, ctypes._SimpleCData), "Expected a simple data ctype"
        size_bytes = reduce(operator.mul, shape, ctypes.sizeof(ctype))
        buf = self.fn_malloc(size_bytes)
        assert buf, "Failed to allocate memory"

        rank = len(shape)
        if rank == 0:
            desc = np_mem.make_zero_d_memref_descriptor(ctype)()
            desc.allocated = buf
            desc.aligned = ctypes.cast(buf, ctypes.POINTER(ctype))
            desc.offset = ctypes.c_longlong(0)
            return desc

        desc = np_mem.make_nd_memref_descriptor(rank, ctype)()
        desc.allocated = buf
        desc.aligned = ctypes.cast(buf, ctypes.POINTER(ctype))
        desc.offset = ctypes.c_longlong(0)
        shape_ctype_t = ctypes.c_longlong * rank
        desc.shape = shape_ctype_t(*shape)

        strides = list(accumulate(reversed(shape[1:]), func=operator.mul))
        strides.reverse()
        strides.append(1)
        desc.strides = shape_ctype_t(*strides)
        return desc

    def dealloc(self, memref_desc: ctypes.Structure) -> None:
        """
        Free underlying memory buffer.

        Args:
            memref_desc: An MLIR memref descriptor.
        """
        # TODO: Expose upstream MemrefDescriptor classes for easier handling
        assert memref_desc.__class__.__name__ == "MemRefDescriptor" or isinstance(
            memref_desc, np_mem.UnrankedMemRefDescriptor
        ), "Invalid memref descriptor"

        if isinstance(memref_desc, np_mem.UnrankedMemRefDescriptor):
            # Unranked memref holds the underlying descriptor as an opaque pointer.
            # Cast the descriptor to a zero ranked memref with an arbitrary type to
            # access the base allocated memory pointer.
            ranked_desc_type = np_mem.make_zero_d_memref_descriptor(ctypes.c_char)
            ranked_desc = ctypes.cast(
                memref_desc.descriptor, ctypes.POINTER(ranked_desc_type)
            )
            memref_desc = ranked_desc[0]

        alloc_ptr = memref_desc.allocated
        if alloc_ptr == 0:
            return

        c_ptr = ctypes.cast(alloc_ptr, ctypes.c_void_p)
        self.fn_free(c_ptr)
        memref_desc.allocated = 0

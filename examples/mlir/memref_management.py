# RUN: %PYTHON %s

import torch
import ctypes

from mlir import ir
from mlir.dialects import func, memref
from mlir.runtime import np_to_memref
from mlir.passmanager import PassManager

from lighthouse import runtime as lh_runtime
from lighthouse import utils as lh_utils
from lighthouse.workload import get_engine


def create_mlir_module(shape: list[int]) -> ir.Module:
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        mem_type = ir.MemRefType.get(shape, ir.F32Type.get())

        # Return a new buffer initialized with input's data.
        @func.func(mem_type)
        def copy(input):
            new_buf = memref.alloc(mem_type, [], [])
            memref.copy(input, new_buf)
            return new_buf

        # Free given buffer.
        @func.func(mem_type)
        def module_dealloc(input):
            memref.dealloc(input)

    return module


def lower_to_llvm(operation: ir.Operation) -> None:
    pm = PassManager("builtin.module")
    pm.add("func.func(llvm-request-c-wrappers)")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")
    pm.add("cse")
    pm.add("canonicalize")
    pm.run(operation)


def main():
    # Validate basic functionality.
    print("Testing memref allocator...")
    mem = lh_runtime.MemRefManager()
    # Check allocation.
    buf = mem.alloc(32, 8, 16, ctype=ctypes.c_float)
    assert buf.allocated != 0, "Invalid allocation"
    assert list(buf.shape) == [32, 8, 16], "Invalid shape"
    assert list(buf.strides) == [128, 16, 1], "Invalid strides"
    # Check deallocation.
    mem.dealloc(buf)
    assert buf.allocated == 0, "Failed deallocation"
    # Double free must not crash.
    mem.dealloc(buf)

    # Zero rank buffer.
    buf = mem.alloc(ctype=ctypes.c_float)
    mem.dealloc(buf)
    # Small buffer.
    buf = mem.alloc(8, ctype=ctypes.c_int8)
    mem.dealloc(buf)
    # Large buffer.
    buf = mem.alloc(1024, 1024, ctype=ctypes.c_int32)
    mem.dealloc(buf)

    # Validate functionality across Python-MLIR boundary.
    print("Testing JIT module memory management...")
    # Buffer shape for testing.
    shape = [16, 32]

    # Create and compile test module.
    kernel = create_mlir_module(shape)
    lower_to_llvm(kernel.operation)
    eng = get_engine(kernel)

    # Validate passing memrefs between Python and jitted module.
    print("...copy test...")
    fn_copy = eng.lookup("copy")

    # Alloc buffer in Python and initialize it.
    in_mem = mem.alloc(*shape, ctype=ctypes.c_float)
    in_np = np_to_memref.ranked_memref_to_numpy([in_mem])
    assert not in_np.flags.owndata, "Expected non-owning memref conversion"
    in_tensor = torch.from_numpy(in_np)
    torch.randn(in_tensor.shape, out=in_tensor)

    out_mem = np_to_memref.make_nd_memref_descriptor(in_tensor.dim(), ctypes.c_float)()
    out_mem.allocated = 0

    args = lh_utils.memrefs_to_packed_args([out_mem, in_mem])
    fn_copy(args)
    assert out_mem.allocated != 0, "Invalid buffer returned"

    out_tensor = torch.from_numpy(np_to_memref.ranked_memref_to_numpy([out_mem]))
    torch.testing.assert_close(out_tensor, in_tensor)

    mem.dealloc(out_mem)
    assert out_mem.allocated == 0, "Failed to dealloc returned buffer"
    mem.dealloc(in_mem)

    # Validate external allocation with deallocation from within jitted module.
    print("...dealloc test...")
    fn_mlir_dealloc = eng.lookup("module_dealloc")
    buf_mem = mem.alloc(*shape, ctype=ctypes.c_float)
    fn_mlir_dealloc(lh_utils.memrefs_to_packed_args([buf_mem]))

    print("SUCCESS")


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        main()

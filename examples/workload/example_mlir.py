"""
Workload example: Element-wise sum of two (M, N) float32 arrays on CPU.

In this example, allocation and deallocation of input arrays is done in MLIR.
"""

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import (
    ranked_memref_to_numpy,
    make_nd_memref_descriptor,
    as_ctype,
)
from mlir.dialects import func, linalg, arith, memref
import ctypes
from contextlib import contextmanager
from lighthouse.utils import get_packed_arg
from lighthouse.utils.execution import (
    lower_payload,
    execute,
    benchmark,
)
from example import ElementwiseSum


def emit_host_alloc(mod, suffix, element_type, rank=2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    index_t = ir.IndexType.get()
    i32_t = ir.IntegerType.get_signless(32)
    with ir.InsertionPoint(mod.body):
        f = func.FuncOp("host_alloc_" + suffix, (rank * (i32_t,), (memref_dyn_t,)))
        f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(f.add_entry_block()):
        dims = [arith.IndexCastOp(index_t, a) for a in list(f.arguments)]
        alloc = memref.alloc(memref_dyn_t, dims, [])
        func.ReturnOp((alloc,))


def emit_host_dealloc(mod, suffix, element_type, rank=2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    with ir.InsertionPoint(mod.body):
        f = func.FuncOp("host_dealloc_" + suffix, ((memref_dyn_t,), ()))
        f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(f.add_entry_block()):
        memref.dealloc(f.arguments[0])
        func.ReturnOp(())


def emit_fill_constant(mod, suffix, value, element_type, rank=2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    with ir.InsertionPoint(mod.body):
        f = func.FuncOp("host_fill_constant_" + suffix, ((memref_dyn_t,), ()))
        f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(f.add_entry_block()):
        const = arith.constant(element_type, value)
        linalg.fill(const, outs=[f.arguments[0]])
        func.ReturnOp(())


def emit_fill_random(mod, suffix, element_type, min=0.0, max=1.0, seed=2):
    rank = 2
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    i32_t = ir.IntegerType.get_signless(32)
    f64_t = ir.F64Type.get()
    with ir.InsertionPoint(mod.body):
        f = func.FuncOp("host_fill_random_" + suffix, ((memref_dyn_t,), ()))
        f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    with ir.InsertionPoint(f.add_entry_block()):
        min_cst = arith.constant(f64_t, min)
        max_cst = arith.constant(f64_t, max)
        seed_cst = arith.constant(i32_t, seed)
        linalg.fill_rng_2d(min_cst, max_cst, seed_cst, outs=[f.arguments[0]])
        func.ReturnOp(())


class ElementwiseSumMLIRAlloc(ElementwiseSum):
    """
    Computes element-wise sum of (M, N) float32 arrays on CPU.

    Extends ElementwiseSum by allocating input arrays in MLIR.
    """

    def __init__(self, M, N):
        super().__init__(M, N)
        # keep track of allocated memrefs
        self.memrefs = {}

    def _allocate_array(self, name, execution_engine):
        if name in self.memrefs:
            return self.memrefs[name]
        alloc_func = execution_engine.lookup("host_alloc_f32")
        shape = (self.M, self.N)
        mref = make_nd_memref_descriptor(len(shape), as_ctype(self.dtype))()
        ptr_mref = ctypes.pointer(ctypes.pointer(mref))
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        alloc_func(get_packed_arg([ptr_mref, *ptr_dims]))
        self.memrefs[name] = mref
        return mref

    def _allocate_inputs(self, execution_engine):
        self._allocate_array("A", execution_engine)
        self._allocate_array("B", execution_engine)
        self._allocate_array("C", execution_engine)

    def _deallocate_all(self, execution_engine):
        for mref in self.memrefs.values():
            dealloc_func = execution_engine.lookup("host_dealloc_f32")
            ptr_mref = ctypes.pointer(ctypes.pointer(mref))
            dealloc_func(get_packed_arg([ptr_mref]))
        self.memrefs = {}

    @contextmanager
    def allocate(self, execution_engine):
        try:
            self._allocate_inputs(execution_engine)
            yield None
        finally:
            self._deallocate_all(execution_engine)

    def get_input_arrays(self, execution_engine):
        A = self._allocate_array("A", execution_engine)
        B = self._allocate_array("B", execution_engine)
        C = self._allocate_array("C", execution_engine)

        # initialize with MLIR
        fill_zero_func = execution_engine.lookup("host_fill_constant_zero_f32")
        fill_random_func = execution_engine.lookup("host_fill_random_f32")
        fill_zero_func(get_packed_arg([ctypes.pointer(ctypes.pointer(C))]))
        fill_random_func(get_packed_arg([ctypes.pointer(ctypes.pointer(A))]))
        fill_random_func(get_packed_arg([ctypes.pointer(ctypes.pointer(B))]))

        return [A, B, C]

    def verify(self, execution_engine, verbose: int = 0) -> bool:
        # compute reference solution with numpy
        A = ranked_memref_to_numpy([self.memrefs["A"]])
        B = ranked_memref_to_numpy([self.memrefs["B"]])
        C = ranked_memref_to_numpy([self.memrefs["C"]])
        C_ref = A + B
        if verbose > 1:
            print("Reference solution:")
            print(C_ref)
            print("Computed solution:")
            print(C)
        success = np.allclose(C, C_ref)

        # Alternatively we could have done the verification in MLIR by emitting
        # a check function.
        # Here we just call the payload function again.
        # self._allocate_array("C_ref", execution_engine)
        # func = execution_engine.lookup("payload")
        # func(get_packed_arg([
        #     ctypes.pointer(ctypes.pointer(self.memrefs["A"])),
        #     ctypes.pointer(ctypes.pointer(self.memrefs["B"])),
        #     ctypes.pointer(ctypes.pointer(self.memrefs["C_ref"])),
        # ]))
        # Check correctness with numpy.
        # C = ranked_memref_to_numpy([self.memrefs["C"]])
        # C_ref = ranked_memref_to_numpy([self.memrefs["C_ref"]])
        # success = np.allclose(C, C_ref)

        if verbose:
            if success:
                print("PASSED")
            else:
                print("FAILED Result mismatch!")
        return success

    def payload_module(self):
        mod = super().payload_module()
        # extend the payload module with de/alloc/fill functions
        with self.context, self.location:
            float32_t = ir.F32Type.get()
            emit_host_alloc(mod, "f32", float32_t)
            emit_host_dealloc(mod, "f32", float32_t)
            emit_fill_constant(mod, "zero_f32", 0.0, float32_t)
            emit_fill_random(mod, "f32", float32_t, min=-1.0, max=1.0)
        return mod


if __name__ == "__main__":
    wload = ElementwiseSumMLIRAlloc(400, 400)

    print(" Dump kernel ".center(60, "-"))
    lower_payload(wload, dump_kernel="bufferized", dump_schedule=False)

    print(" Execute ".center(60, "-"))
    execute(wload, verbose=2)

    print(" Benchmark ".center(60, "-"))
    times = benchmark(wload)
    times *= 1e6  # convert to microseconds
    # compute statistics
    mean = np.mean(times)
    min = np.min(times)
    max = np.max(times)
    std = np.std(times)
    print(f"Timings (us): mean={mean:.2f}+/-{std:.2f} min={min:.2f} max={max:.2f}")
    flop_count = wload.get_complexity()[0]
    gflops = flop_count / (mean * 1e-6) / 1e9
    print(f"Throughput: {gflops:.2f} GFLOPS")

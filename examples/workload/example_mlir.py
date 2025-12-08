# RUN: %PYTHON %s  | FileCheck %s
# CHECK: func.func @payload
# CHECK: PASSED
# CHECK: Throughput:
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
from mlir.execution_engine import ExecutionEngine
import ctypes
from contextlib import contextmanager
from lighthouse.utils import (
    get_packed_arg,
    memrefs_to_packed_args,
    memref_to_ctype,
)
from example import ElementwiseSum
from lighthouse.workload import (
    execute,
    benchmark,
)


def emit_host_alloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    index_t = ir.IndexType.get()
    i32_t = ir.IntegerType.get_signless(32)
    inputs = rank * (i32_t,)

    @func.func(*inputs, name="host_alloc_" + suffix)
    def alloc_func(*shape):
        dims = [arith.index_cast(index_t, a) for a in shape]
        alloc = memref.alloc(memref_dyn_t, dims, [])
        return alloc

    alloc_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_host_dealloc(suffix: str, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func.func(memref_dyn_t, name="host_dealloc_" + suffix)
    def dealloc_func(buffer):
        memref.dealloc(buffer)

    dealloc_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_fill_constant(suffix: str, value: float, element_type: ir.Type, rank: int = 2):
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)

    @func.func(memref_dyn_t, name="host_fill_constant_" + suffix)
    def init_func(buffer):
        const = arith.constant(element_type, value)
        linalg.fill(const, outs=[buffer])

    init_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def emit_fill_random(
    suffix: str,
    element_type: ir.Type,
    min: float = 0.0,
    max: float = 1.0,
    seed: int = 2,
):
    rank = 2
    dyn = ir.ShapedType.get_dynamic_size()
    memref_dyn_t = ir.MemRefType.get(rank * (dyn,), element_type)
    i32_t = ir.IntegerType.get_signless(32)
    f64_t = ir.F64Type.get()

    @func.func(memref_dyn_t, name="host_fill_random_" + suffix)
    def init_func(buffer):
        min_cst = arith.constant(f64_t, min)
        max_cst = arith.constant(f64_t, max)
        seed_cst = arith.constant(i32_t, seed)
        linalg.fill_rng_2d(min_cst, max_cst, seed_cst, outs=[buffer])

    init_func.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


class ElementwiseSumMLIRAlloc(ElementwiseSum):
    """
    Computes element-wise sum of (M, N) float32 arrays on CPU.

    Extends ElementwiseSum by allocating input arrays in MLIR.
    """

    def __init__(self, M: int, N: int):
        super().__init__(M, N)
        # keep track of allocated memrefs
        self.memrefs = {}

    def _allocate_array(
        self, name: str, execution_engine: ExecutionEngine
    ) -> ctypes.Structure:
        if name in self.memrefs:
            return self.memrefs[name]
        alloc_func = execution_engine.lookup("host_alloc_f32")
        # construct a memref descriptor for the result memref
        shape = (self.M, self.N)
        mref = make_nd_memref_descriptor(len(shape), as_ctype(self.dtype))()
        ptr_mref = memref_to_ctype(mref)
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        alloc_func(get_packed_arg([ptr_mref, *ptr_dims]))
        self.memrefs[name] = mref
        return mref

    def _deallocate_all(self, execution_engine: ExecutionEngine):
        for mref in self.memrefs.values():
            dealloc_func = execution_engine.lookup("host_dealloc_f32")
            dealloc_func(memrefs_to_packed_args([mref]))
        self.memrefs = {}

    def get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        A = self._allocate_array("A", execution_engine)
        B = self._allocate_array("B", execution_engine)
        C = self._allocate_array("C", execution_engine)

        # initialize with MLIR
        fill_zero_func = execution_engine.lookup("host_fill_constant_zero_f32")
        fill_random_func = execution_engine.lookup("host_fill_random_f32")
        fill_zero_func(memrefs_to_packed_args([C]))
        fill_random_func(memrefs_to_packed_args([A]))
        fill_random_func(memrefs_to_packed_args([B]))

        return [A, B, C]

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            yield self.get_input_arrays(execution_engine)
        finally:
            self._deallocate_all(execution_engine)

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
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
        # func(memrefs_to_packed_args([
        #     self.memrefs["A"],
        #     self.memrefs["B"],
        #     self.memrefs["C_ref"],
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
        with ir.InsertionPoint(mod.body):
            float32_t = ir.F32Type.get()
            emit_host_alloc("f32", float32_t)
            emit_host_dealloc("f32", float32_t)
            emit_fill_constant("zero_f32", 0.0, float32_t)
            emit_fill_random("f32", float32_t, min=-1.0, max=1.0)
        return mod


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        wload = ElementwiseSumMLIRAlloc(400, 400)

        print(" Dump kernel ".center(60, "-"))
        wload.lower_payload(dump_payload="bufferized", dump_schedule=False)

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

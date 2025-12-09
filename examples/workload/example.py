# RUN: %PYTHON %s | FileCheck %s
# CHECK: func.func @payload
# CHECK: PASSED
# CHECK: Throughput:
"""
Workload example: Element-wise sum of two (M, N) float32 arrays on CPU.
"""

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir.dialects import func, linalg, bufferization
from mlir.dialects import transform
from mlir.execution_engine import ExecutionEngine
from contextlib import contextmanager
from functools import cached_property
import ctypes
from typing import Optional
from lighthouse.utils.mlir import (
    apply_registered_pass,
    canonicalize,
    match,
)
from lighthouse.workload import (
    Workload,
    execute,
    benchmark,
)


class ElementwiseSum(Workload):
    """
    Computes element-wise sum of (M, N) float32 arrays on CPU.

    We can construct the input arrays and compute the reference solution in
    Python with Numpy.

    We use @cached_property to store the inputs and reference solution in the
    object so that they are only computed once.
    """

    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self.dtype = np.float32

    @cached_property
    def _input_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(" * Generating input arrays...")
        np.random.seed(2)
        A = np.random.rand(self.M, self.N).astype(self.dtype)
        B = np.random.rand(self.M, self.N).astype(self.dtype)
        C = np.zeros((self.M, self.N), dtype=self.dtype)
        return [A, B, C]

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        print(" * Computing reference solution...")
        A, B, _ = self._input_arrays
        return A + B

    def _get_input_arrays(self) -> list[ctypes.Structure]:
        return [get_ranked_memref_descriptor(a) for a in self._input_arrays]

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            yield self._get_input_arrays()
        finally:
            # cached numpy arrays are deallocated automatically
            pass

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        C = self._input_arrays[2]
        C_ref = self._reference_solution
        if verbose > 1:
            print("Reference solution:")
            print(C_ref)
            print("Computed solution:")
            print(C)
        success = np.allclose(C, C_ref)
        if verbose:
            if success:
                print("PASSED")
            else:
                print("FAILED Result mismatch!")
        return success

    def shared_libs(self) -> list[str]:
        return []

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        flop_count = self.M * self.N  # one addition per element
        memory_reads = 2 * self.M * self.N * nbytes  # read A and B
        memory_writes = self.M * self.N * nbytes  # write C
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        mod = ir.Module.create()

        with ir.InsertionPoint(mod.body):
            float32_t = ir.F32Type.get()
            shape = (self.M, self.N)
            tensor_t = ir.RankedTensorType.get(shape, float32_t)
            memref_t = ir.MemRefType.get(shape, float32_t)
            fargs = [memref_t, memref_t, memref_t]

            @func.func(*fargs, name=self.payload_function_name)
            def payload(A, B, C):
                a_tensor = bufferization.to_tensor(tensor_t, A, restrict=True)
                b_tensor = bufferization.to_tensor(tensor_t, B, restrict=True)
                c_tensor = bufferization.to_tensor(
                    tensor_t, C, restrict=True, writable=True
                )
                add = linalg.add(a_tensor, b_tensor, outs=[c_tensor])
                bufferization.materialize_in_destination(
                    None, add, C, restrict=True, writable=True
                )

        payload.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return mod

    def schedule_module(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> ir.Module:
        schedule_module = ir.Module.create()
        schedule_module.operation.attributes["transform.with_named_sequence"] = (
            ir.UnitAttr.get()
        )
        with ir.InsertionPoint(schedule_module.body):
            named_sequence = transform.named_sequence(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
            )
            with ir.InsertionPoint(named_sequence.body):
                anytype = transform.AnyOpType.get()
                func = match(named_sequence.bodyTarget, ops={"func.func"})
                mod = transform.get_parent_op(
                    anytype,
                    func,
                    op_name="builtin.module",
                    deduplicate=True,
                )
                mod = apply_registered_pass(mod, "one-shot-bufferize")
                mod = apply_registered_pass(mod, "convert-linalg-to-loops")
                transform.apply_cse(mod)
                canonicalize(mod)

                if stop_at_stage == "bufferized":
                    transform.YieldOp()
                    return schedule_module

                mod = apply_registered_pass(mod, "convert-scf-to-cf")
                mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
                mod = apply_registered_pass(mod, "convert-cf-to-llvm")
                mod = apply_registered_pass(mod, "convert-arith-to-llvm")
                mod = apply_registered_pass(mod, "convert-func-to-llvm")
                mod = apply_registered_pass(mod, "reconcile-unrealized-casts")
                transform.YieldOp()

        return schedule_module


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        wload = ElementwiseSum(400, 400)

        print(" Dump kernel ".center(60, "-"))
        wload.lower_payload(dump_payload="bufferized", dump_schedule=True)

        print(" Execute 1 ".center(60, "-"))
        execute(wload, verbose=2)

        print(" Execute 2 ".center(60, "-"))
        execute(wload, verbose=1)

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

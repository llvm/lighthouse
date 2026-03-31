# RUN: %PYTHON %s | FileCheck %s
# CHECK: func.func @payload
# CHECK: PASSED
# CHECK: Throughput:
"""
Kernel execution example: Element-wise sum of two (M, N) float32 arrays on CPU.
"""

import ctypes
from functools import cached_property
from typing import Optional

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from mlir.dialects import func, linalg, bufferization
from mlir.dialects import transform

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.helper import match
from lighthouse.pipeline.stage import PassBundles, apply_bundle
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import get_bench_wrapper_schedule


class ElementwiseSum:
    """
    Computes element-wise sum of (M, N) float32 arrays on CPU.

    We can construct the input arrays and compute the reference solution in
    Python with Numpy.

    We use @cached_property to store the inputs and reference solution in the
    object so that they are only computed once.
    """

    payload_function_name: str = "payload"

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

    def _get_input_arrays(self) -> list[ctypes.Structure]:
        return [get_ranked_memref_descriptor(a) for a in self._input_arrays]

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

    def schedule_modules(
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
                mod = apply_bundle(mod, PassBundles["BufferizationBundle"])
                mod = apply_bundle(mod, PassBundles["MLIRLoweringBundle"])
                mod = apply_bundle(mod, PassBundles["CleanupBundle"])

                if stop_at_stage == "bufferized":
                    transform.YieldOp()
                    return [schedule_module]

                mod = apply_bundle(mod, PassBundles["LLVMLoweringBundle"])
                transform.YieldOp()

        return [
            get_bench_wrapper_schedule(self.payload_function_name),
            schedule_module,
        ]


if __name__ == "__main__":
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = ElementwiseSum(400, 400)

        print(" Dump kernel ".center(60, "-"))
        pipeline = TransformDriver(wload.schedule_modules(stop_at_stage="bufferized"))
        payload = pipeline.apply(wload.payload_module())
        print(payload)
        for schedule_module in wload.schedule_modules():
            print(schedule_module)

        pipeline = TransformDriver(wload.schedule_modules())
        payload = pipeline.apply(wload.payload_module())
        runner = Runner(shared_libs=wload.shared_libs())

        print(" Execute 1 ".center(60, "-"))
        runner.execute(
            payload,
            host_input_buffers=wload._input_arrays,
            payload_function_name=wload.payload_function_name,
        )

        print(" Execute 2 ".center(60, "-"))
        runner.execute(
            payload,
            host_input_buffers=wload._input_arrays,
            payload_function_name=wload.payload_function_name,
        )

        print(" Correctness test ".center(60, "-"))
        A, B, C = wload._input_arrays
        C_ref = A + B
        print("Reference solution:")
        print(C_ref)
        print("Computed solution:")
        print(C)
        success = np.allclose(C, C_ref)
        if success:
            print("PASSED")
        else:
            print("FAILED Result mismatch!")

        print(" Benchmark ".center(60, "-"))
        times = runner.benchmark(
            payload,
            host_input_buffers=wload._input_arrays,
        )
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

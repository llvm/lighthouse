# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU softmax benchmark.
"""

import argparse
import ctypes
from typing import Optional
from functools import cached_property

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import (
    get_bench_wrapper_schedule,
    GPUMemoryManager,
)
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.ingress.mlir_gen.gpu_softmax_payload import generate_gpu_softmax_payload
from lighthouse.schedule.xegpu.softmax_schedule import get_softmax_schedule_module


def softmax_complexity(M: int, N: int, nbytes: int):
    """
    Complexity of softmax operation.

    For each row:
    - O(N) to find max
    - O(N) to compute exp(x - max) and sum
    - O(N) to normalize
    Total: 3*N operations per row, but with transcendental (exp) operations
    """
    # Approximation: 5 FLOPs per element (max, sub, exp, sum, div)
    flop_count = M * N * 5
    memory_reads = M * N * nbytes  # read input
    memory_writes = M * N * nbytes  # write output
    return flop_count, memory_reads, memory_writes


def check_correctness(
    input_arr: np.ndarray, output_arr: np.ndarray, verbose: int = 0
) -> bool:
    # Use float32 for computation
    x = input_arr.astype(np.float32)
    # Compute softmax along axis 1 (each row independently)
    # Numerically stable version: subtract max before exp
    max_vals = np.max(x, axis=1, keepdims=True)
    exp_vals = np.exp(x - max_vals)
    sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
    output_ref = exp_vals / sum_vals

    output = output_arr.astype(np.float32)

    if verbose > 1:
        print("Reference solution (first 5 rows):")
        print(output_ref[:5])
        print("Computed solution (first 5 rows):")
        print(output[:5])

    # Check row sums are close to 1.0
    row_sums = np.sum(output, axis=1)
    sums_ok = np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6)

    # Check values match reference
    values_ok = np.allclose(output, output_ref, rtol=1e-4, atol=1e-6)

    success = sums_ok and values_ok

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED!")
            if not sums_ok:
                print(
                    f"  Row sums check failed. Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}"
                )
            if not values_ok:
                max_diff = np.abs(output - output_ref).max()
                print(f"  Values mismatch. Max abs diff: {max_diff:.6e}")
    return success


class XeGPUSoftmax:
    """
    Softmax workload on XeGPU.

    Computes softmax along the last dimension (rows):
    output[i, j] = exp(input[i, j] - max_i) / sum_i(exp(input[i, j] - max_i))

    where max_i and sum_i are computed over row i.
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: str = "f32",
    ):
        self.M = M
        self.N = N
        self.shape = (M, N)
        assert dtype == "f32", "Only f32 type is supported for softmax"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray]:
        """Generate initial values on host with numpy."""
        np.random.seed(42)
        # Use values in range [-0.5, 0.5] to avoid numerical issues
        input_arr = np.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, input_arr)

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        return softmax_complexity(self.M, self.N, nbytes)

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for softmax payload."""
        return generate_gpu_softmax_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
        )

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        """Generate transform schedule for softmax."""
        return [
            get_bench_wrapper_schedule(self.payload_function_name),
            get_softmax_schedule_module(
                stop_at_stage=stop_at_stage,
                parameters=parameters,
            ),
        ]

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Softmax using MLIR XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=2,
        default=[1024, 64],
        help="M,N matrix sizes (MxN)",
    )
    parser.add_argument(
        "--wg-rows",
        type=int,
        default=64,
        help="Number of rows per workgroup.",
    )
    parser.add_argument(
        "--sg-rows",
        type=int,
        default=8,
        help="Number of rows per subgroup.",
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=16,
        help="Subgroup size.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=1000,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=20,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the softmax computation.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Dump kernel IR at different stages of lowering and exit without "
        "executing the kernel.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity (e.g. print reference and computed solutions).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()

    params = {
        "sizes": args.sizes,
        "wg_rows": args.wg_rows,
        "sg_rows": args.sg_rows,
        "subgroup_size": args.subgroup_size,
    }

    M, N = args.sizes
    dtype = "f32"

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        wload = XeGPUSoftmax(M=M, N=N, dtype=dtype)

        if args.dump_kernel or args.dump_schedule:
            pipeline = TransformDriver(
                wload.schedule_modules(
                    stop_at_stage=args.dump_kernel, parameters=params
                )
            )
            payload = pipeline.apply(wload.payload_module())
            if args.dump_kernel:
                print(payload)
            if args.dump_schedule:
                for schedule_module in wload.schedule_modules(parameters=params):
                    print(schedule_module)
        else:
            pipeline = TransformDriver(wload.schedule_modules(parameters=params))
            payload = pipeline.apply(wload.payload_module())
            runner = Runner(shared_libs=wload.shared_libs())
            if args.check_result:
                # Setup callback function to copy result from device to host.
                result_host_copy = np.zeros(wload.shape, dtype=wload.dtype)

                def argument_access_callback(
                    inputs: list[ctypes.Structure],
                    *,
                    memory_manager: GPUMemoryManager,
                    **kwargs,
                ):
                    memory_manager.copy(inputs[0], result_host_copy)

                # Execute kernel once.
                runner.execute(
                    payload,
                    host_input_buffers=wload._initial_host_arrays,
                    mem_manager_cls=wload.memory_manager_class,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )

                # Compute reference solution on host.
                success = check_correctness(
                    wload._initial_host_arrays[1],
                    result_host_copy,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")

            times = runner.benchmark(
                payload,
                host_input_buffers=wload._initial_host_arrays,
                mem_manager_cls=wload.memory_manager_class,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            print(
                f"sizes={list2str(args.sizes)} "
                f"dt={dtype} "
                f"wg-rows={args.wg_rows} "
                f"sg-rows={args.sg_rows} "
                f"subgroup-size={args.subgroup_size} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f} "
            )

# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU layer_norm benchmark.
"""

import argparse
from typing import Optional
from functools import cached_property

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import GPUMemoryManager
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.ingress.mlir_gen import get_mlir_elem_type
from lighthouse.ingress.mlir_gen.gpu_layer_norm_payload import (
    generate_gpu_layer_norm_payload,
)
from lighthouse.schedule.xegpu import layer_norm_schedule, xegpu_to_binary


def layer_norm_complexity(M: int, N: int, nbytes: int):
    """
    Complexity of layer_norm operation.

    Per row of length N:
    - N adds for mean reduction
    - N subs + N muls + N adds for variance reduction
    - N subs + N muls (inv_std) + N muls (gamma) + N adds (beta) + 1 rsqrt
    Total ~ 8 FLOPs per element.
    """
    flop_count = M * N * 8
    memory_reads = M * N * nbytes + 2 * N * nbytes  # input + gamma + beta
    memory_writes = M * N * nbytes
    return flop_count, memory_reads, memory_writes


def check_correctness(
    input_arr: np.ndarray,
    gamma_arr: np.ndarray,
    beta_arr: np.ndarray,
    output_arr: np.ndarray,
    eps: float,
    verbose: int = 0,
) -> bool:
    x = input_arr.astype(np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    output_ref = (x - mean) * inv_std * gamma_arr.astype(np.float32) + beta_arr.astype(
        np.float32
    )

    output = output_arr.astype(np.float32)

    if verbose > 1:
        print("Reference solution (first 5 rows):")
        print(output_ref[:5])
        print("Computed solution (first 5 rows):")
        print(output[:5])

    values_ok = np.allclose(output, output_ref, rtol=1e-3, atol=1e-4)

    if verbose:
        if values_ok:
            print("PASSED")
        else:
            max_diff = np.abs(output - output_ref).max()
            print(f"FAILED! Max abs diff: {max_diff:.6e}")
    return values_ok


class XeGPULayerNorm:
    """
    Layer norm workload on XeGPU.

    Computes layer normalization along the last dimension (rows):
        mean_i    = (1/N) * sum_j x[i, j]
        var_i     = (1/N) * sum_j (x[i, j] - mean_i)^2
        out[i, j] = (x[i, j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]
    """

    def __init__(
        self,
        M: int,
        N: int,
        dtype: str = "f32",
        eps: float = 1e-5,
    ):
        self.M = M
        self.N = N
        self.eps = eps
        self.shape = (M, N)
        self.bias_shape = (N,)
        assert dtype == "f32", "Only f32 type is supported for layer_norm"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray]:
        """Generate initial values on host with numpy."""
        np.random.seed(42)
        input_arr = np.random.uniform(-1.0, 1.0, self.shape).astype(self.dtype)
        gamma_arr = np.random.uniform(0.5, 1.5, self.bias_shape).astype(self.dtype)
        beta_arr = np.random.uniform(-0.1, 0.1, self.bias_shape).astype(self.dtype)
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, input_arr, gamma_arr, beta_arr)

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        return layer_norm_complexity(self.M, self.N, nbytes)

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for layer_norm payload."""
        mod = generate_gpu_layer_norm_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            dtype=self.elem_type,
            eps=self.eps,
        )
        # Emit the memory management utility functions into the payload
        # module: the 2D input/output and the 1D gamma/beta bias vectors.
        ranks_and_types = [(2, self.elem_type), (1, self.elem_type)]
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        """Generate transform schedule for layer_norm."""
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            layer_norm_schedule(
                stop_at_stage=stop_at_stage,
                parameters=parameters,
            )
        )

        if stop_at_stage and stop_at_stage != "final":
            return schedules

        schedules.append(xegpu_to_binary())

        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="LayerNorm using MLIR XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=2,
        default=[1024, 512],
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
        "--reduction-step-size",
        type=int,
        default=16,
        help="Step size for reduction loop tiling.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="Epsilon added to variance for numerical stability.",
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
        help="Check the result of the layer_norm computation.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "gpu-outlining",
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
        help="Increase output verbosity.",
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
        "reduction_step_size": args.reduction_step_size,
    }

    M, N = args.sizes
    dtype = "f32"

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        wload = XeGPULayerNorm(M=M, N=N, dtype=dtype, eps=args.eps)

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
            runner = Runner(
                payload,
                mem_manager_cls=wload.memory_manager_class,
                shared_libs=wload.shared_libs(),
            )
            if args.check_result:
                result_host_copy = np.zeros(wload.shape, dtype=wload.dtype)
                argument_access_callback = Runner.get_gpu_argument_access_callback(
                    result_host_copy, arg_index=0
                )

                runner.execute(
                    host_input_buffers=wload._initial_host_arrays,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )

                _, input_arr, gamma_arr, beta_arr = wload._initial_host_arrays
                success = check_correctness(
                    input_arr,
                    gamma_arr,
                    beta_arr,
                    result_host_copy,
                    eps=wload.eps,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")
                else:
                    print("Result is correct. Proceeding to benchmark...")

            times = runner.benchmark(
                host_input_buffers=wload._initial_host_arrays,
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

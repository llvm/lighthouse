# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU fused attention benchmark.
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
from lighthouse.ingress.mlir_gen.gpu_attention_payload import (
    generate_gpu_attention_payload,
)
from lighthouse.schedule.xegpu import fused_attention_schedule, xegpu_to_binary


def fused_attention_complexity(Z: int, H: int, n_ctx: int, n_head: int, nbytes: int):
    """
    Complexity of fused attention operation.

    For each batch and head:
    - Q @ K^T: O(n_ctx^2 * n_head) operations
    - Softmax: O(n_ctx^2) operations
    - Attention @ V: O(n_ctx^2 * n_head) operations
    Total: approximately 2*n_ctx^2*n_head FLOPs per batch and head
    """
    # Approximation: 2 * n_ctx^2 * n_head FLOPs per batch and head
    flop_count = Z * H * 2 * n_ctx * n_ctx * n_head
    # Memory: read Q, K, V and write output
    memory_reads = 3 * Z * H * n_ctx * n_head * nbytes
    memory_writes = Z * H * n_ctx * n_head * nbytes
    return flop_count, memory_reads, memory_writes


def check_correctness(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    output_arr: np.ndarray,
    verbose: int = 0,
) -> bool:
    """
    Check correctness of fused attention output.

    Reference implementation:
    - scores = Q @ K^T / sqrt(n_head)
    - attention_weights = softmax(scores, dim=-1)
    - output = attention_weights @ V
    """
    # Use float32 for computation
    Q_f32 = Q.astype(np.float32)
    K_f32 = K.astype(np.float32)
    V_f32 = V.astype(np.float32)

    Z, H, n_ctx, n_head = Q.shape
    scale = 1.0 / np.sqrt(n_head)

    output_ref = np.zeros_like(Q_f32)

    # Compute reference for each batch and head
    for z in range(Z):
        for h in range(H):
            # scores = Q @ K^T / sqrt(n_head)
            scores = Q_f32[z, h] @ K_f32[z, h].T * scale

            # softmax along last dimension
            max_vals = np.max(scores, axis=1, keepdims=True)
            exp_vals = np.exp(scores - max_vals)
            sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
            attention_weights = exp_vals / sum_vals

            # output = attention_weights @ V
            output_ref[z, h] = attention_weights @ V_f32[z, h]

    output = output_arr.astype(np.float32)

    if verbose > 1:
        print("Reference solution (first batch, first head, first 5 rows):")
        print(output_ref[0, 0, :5])
        print("Computed solution (first batch, first head, first 5 rows):")
        print(output[0, 0, :5])

    # Check values match reference
    values_ok = np.allclose(output, output_ref, rtol=1e-3, atol=1e-3)
    success = values_ok

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED!")
            if not values_ok:
                max_diff = np.abs(output - output_ref).max()
                print(f"  Values mismatch. Max abs diff: {max_diff:.6e}")
    return success


class XeGPUFusedAttention:
    """
    Fused attention workload on XeGPU. This workload starts with standard attention
    at linalg level and applies a series of transformations to arrive at a fused
    attention kernel where each work group computes a tile of the output with the
    fused attention algorithm.

    Computes fused attention:
    output = softmax(Q @ K^T / sqrt(n_head)) @ V

    All Q, K, V matrices have shape (Z, H, n_ctx, n_head) where:
    - Z: batch size
    - H: number of heads
    - n_ctx: context length
    - n_head: head dimension
    """

    def __init__(
        self,
        Z: int,
        H: int,
        n_ctx: int,
        n_head: int,
        dtype: str = "f16",
    ):
        self.Z = Z
        self.H = H
        self.n_ctx = n_ctx
        self.n_head = n_head
        self.shape = (Z, H, n_ctx, n_head)
        assert dtype == "f16", "Only f16 type is supported for fused attention"
        self.elem_type = get_mlir_elem_type(dtype)
        self.dtype = mlir_to_numpy_dtype(self.elem_type)
        self.memory_manager_class = GPUMemoryManager
        self.payload_function_name = "payload"

    @cached_property
    def _initial_host_arrays(self) -> tuple[np.ndarray]:
        """Generate initial values on host with numpy."""
        np.random.seed(42)
        # Initialize Q, K, V with small random values
        Q = np.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        K = np.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        V = np.random.uniform(-0.5, 0.5, self.shape).astype(self.dtype)
        output_arr = np.zeros(self.shape, dtype=self.dtype)
        return (output_arr, Q, K, V)

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes = np.dtype(self.dtype).itemsize
        return fused_attention_complexity(
            self.Z, self.H, self.n_ctx, self.n_head, nbytes
        )

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for fused attention payload."""
        mod = generate_gpu_attention_payload(
            func_name=self.payload_function_name,
            Z=self.Z,
            H=self.H,
            n_ctx=self.n_ctx,
            n_head=self.n_head,
            dtype=self.elem_type,
        )
        ranks_and_types = [(4, self.elem_type)]
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        """Generate transform schedule for fused attention."""
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            fused_attention_schedule(
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
        description="Fused Attention using MLIR XeGPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (Z)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (H)",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context length (sequence length)",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=64,
        help="Head dimension",
    )
    parser.add_argument(
        "--wg-rows",
        type=int,
        default=128,
        help="Number of Q*K^T*V rows computed by each work group",
    )
    parser.add_argument(
        "--sg-rows",
        type=int,
        default=16,
        help="Number of Q*K^T*V rows computed by each subgroup",
    )
    parser.add_argument(
        "--subgroup-size",
        type=int,
        default=16,
        help="Subgroup size",
    )
    parser.add_argument(
        "--inner-loop-tile-size",
        type=int,
        default=64,
        help="Tile size for the inner reduction dimension (K/V sequence length)",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=500,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=500,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the fused attention computation.",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        choices=[
            "initial",
            "outer-tiled",
            "inner-tiled",
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
        help="Increase output verbosity (e.g. print reference and computed solutions).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_cli()

    params = {
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "n_ctx": args.n_ctx,
        "n_head": args.n_head,
        "wg_rows": args.wg_rows,
        "sg_rows": args.sg_rows,
        "subgroup_size": args.subgroup_size,
        "inner_loop_tile_size": args.inner_loop_tile_size,
    }

    Z = args.batch_size
    H = args.num_heads
    n_ctx = args.n_ctx
    n_head = args.n_head
    dtype = "f16"

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        wload = XeGPUFusedAttention(Z=Z, H=H, n_ctx=n_ctx, n_head=n_head, dtype=dtype)

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
                # Setup callback function to copy result from device to host.
                result_host_copy = np.zeros(wload.shape, dtype=wload.dtype)
                argument_access_callback = Runner.get_gpu_argument_access_callback(
                    result_host_copy, arg_index=0
                )

                # Execute kernel once.
                runner.execute(
                    host_input_buffers=wload._initial_host_arrays,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )

                # Compute reference solution on host.
                Q, K, V = wload._initial_host_arrays[1:4]
                success = check_correctness(
                    Q,
                    K,
                    V,
                    result_host_copy,
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

            print(
                f"batch-size={Z} "
                f"num-heads={H} "
                f"n-ctx={n_ctx} "
                f"n-head={n_head} "
                f"dt={dtype} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f} "
            )

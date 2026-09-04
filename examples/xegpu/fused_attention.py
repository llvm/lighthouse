# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU fused attention benchmark.
"""

import argparse
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
from lighthouse.schedule.parameters import ScheduleParameters


def fused_attention_complexity(
    batch_size: int, n_head: int, n_ctx: int, d_head: int, nbytes: int
):
    """
    Complexity of fused attention operation.

    Counts the two matmuls only, at 2 FLOPs per multiply-accumulate, which is the
    convention used by the flash attention tutorials (and hence what published
    attention FLOPS numbers can be compared against). For each batch and head:
    - Q @ K^T:        2 * n_ctx^2 * d_head FLOPs
    - Attention @ V:  2 * n_ctx^2 * d_head FLOPs
    The softmax is left out: it is O(n_ctx^2) (~2% of the above at d_head = 64) and
    is not multiply-accumulate work. Halve the total for a causal mask.
    """
    # 2 matmuls, 2 * n_ctx^2 * d_head FLOPs each, per batch and head
    flop_count = batch_size * n_head * 4 * n_ctx * n_ctx * d_head
    # Memory: read Q, K, V and write output
    memory_reads = 3 * batch_size * n_head * n_ctx * d_head * nbytes
    memory_writes = batch_size * n_head * n_ctx * d_head * nbytes
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
    - scores = Q @ K^T / sqrt(d_head)
    - attention_weights = softmax(scores, dim=-1)
    - output = attention_weights @ V
    """
    # Use float32 for computation
    Q_f32 = Q.astype(np.float32)
    K_f32 = K.astype(np.float32)
    V_f32 = V.astype(np.float32)

    batch_size, n_head, n_ctx, d_head = Q.shape
    scale = 1.0 / np.sqrt(d_head)

    output_ref = np.zeros_like(Q_f32)

    # Compute reference for each batch and head
    for z in range(batch_size):
        for h in range(n_head):
            # scores = Q @ K^T / sqrt(d_head)
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
    values_ok = np.allclose(output, output_ref, rtol=1e-3, atol=5e-3)
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
    output = softmax(Q @ K^T / sqrt(d_head)) @ V

    All Q, K, V matrices have shape (batch_size, n_head, n_ctx, d_head) where:

        - batch_size: batch size
        - n_head: number of heads
        - n_ctx: context length
        - d_head: head dimension
    """

    def __init__(
        self,
        batch_size: int,
        n_head: int,
        n_ctx: int,
        d_head: int,
        dtype: str = "f16",
    ):
        self.batch_size = batch_size
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.d_head = d_head
        self.shape = (batch_size, n_head, n_ctx, d_head)
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
            self.batch_size, self.n_head, self.n_ctx, self.d_head, nbytes
        )

    def payload_module(self) -> ir.Module:
        """Generate MLIR module for fused attention payload."""
        mod = generate_gpu_attention_payload(
            func_name=self.payload_function_name,
            batch_size=self.batch_size,
            n_head=self.n_head,
            n_ctx=self.n_ctx,
            d_head=self.d_head,
            dtype=self.elem_type,
        )
        ranks_and_types = [(4, self.elem_type)]
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self,
        stop_at_stage: str | None = None,
        parameters: ScheduleParameters | None = None,
    ) -> list[ir.Module]:
        """Generate transform schedule for fused attention."""
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            fused_attention_schedule(
                stop_at_stage=stop_at_stage,
                params=parameters,
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
        "--n-head",
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
        "--d-head",
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
        "--reduction-tile",
        type=int,
        default=64,
        help="Tile size for the inner reduction dimension (K/V sequence length)",
    )
    parser.add_argument(
        "--reference-flash",
        action="store_true",
        help="Emit the flash loop from the hand-written generator "
        "(transform_ext.replace_with_fused_attention) instead of deriving it from "
        "the payload chain. Kept as a reference point for comparing the two.",
    )
    parser.add_argument(
        "--q-load-tile",
        type=int,
        nargs=2,
        default=[16, 32],
        help="Q load tile size.",
    )
    parser.add_argument(
        "--v-load-tile",
        type=int,
        nargs=2,
        default=[32, 32],
        help="V load tile size.",
    )
    parser.add_argument(
        "--prefetch-tile",
        type=int,
        nargs=2,
        default=[16, 32],
        help="Prefetch tile size for prefetching K and V tiles.",
    )
    parser.add_argument(
        "--nb-prefetch",
        type=int,
        default=1,
        help="Number of K/V tiles to prefetch ahead in the inner loop (0 disables).",
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
            "tiled",
            "reduction-tiled",
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

    layer_params = {
        "layer_kind": "attention",
        "batch_size": args.batch_size,
        "n_head": args.n_head,
        "n_ctx": args.n_ctx,
        "d_head": args.d_head,
        "wg_tile": [1, args.wg_rows],
        "sg_rows": args.sg_rows,
        "subgroup_size": args.subgroup_size,
        "reduction_tile": args.reduction_tile,
        "reference_flash": args.reference_flash,
        "q_load_tile": args.q_load_tile,
        "v_load_tile": args.v_load_tile,
        "prefetch_tile": args.prefetch_tile,
        "nb_prefetch": args.nb_prefetch,
    }

    params = ScheduleParameters([layer_params])

    batch_size = args.batch_size
    n_head = args.n_head
    n_ctx = args.n_ctx
    d_head = args.d_head
    dtype = "f16"

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        wload = XeGPUFusedAttention(
            batch_size=batch_size,
            n_head=n_head,
            n_ctx=n_ctx,
            d_head=d_head,
            dtype=dtype,
        )

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
                f"batch-size={batch_size} "
                f"n-head={n_head} "
                f"n-ctx={n_ctx} "
                f"d-head={d_head} "
                f"dt={dtype} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f} "
            )

# RUN: %PYTHON %s --sizes 512 1024 128 --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --transpose-a | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --transpose-b | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --relu | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias --relu | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --no-accumulate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --truncate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias --truncate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias --relu --no-accumulate-c | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU matrix multiplication example.
"""

import argparse
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, ClassVar

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import (
    MemoryManager,
    GPUMemoryManager,
)
from lighthouse.schedule.xegpu import mlp_schedule, xegpu_to_binary
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.ingress.mlir_gen import generate_gpu_matmul_payload, get_mlir_elem_type
from lighthouse.schedule.xegpu import XeGPUParameterSelector


def matmul_complexity(
    M: int,
    N: int,
    K: int,
    bias: bool,
    relu: bool,
    accumulate_c: bool,
    nbytes_ab: int,
    nbytes_c: int,
):
    """Complexity of matmul operation with optional post-ops"""
    flop_count = 2 * M * N * K
    memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
    memory_writes = M * N * nbytes_c  # write C
    # Below we assume the post-ops are tiled-and-fused and do not cause
    # reads/writes to global memory.
    if bias:
        flop_count += M * N
        memory_reads += N * nbytes_c  # read bias vector
    if relu:
        flop_count += M * N
    if accumulate_c:
        memory_reads += M * N * nbytes_c  # read C for accumulation
    return flop_count, memory_reads, memory_writes


@dataclass
class XeGPUMatMul:
    """
    Matrix multiplication kernel on XeGPU.

    Computes C = A * B for input matrices A (M x K) and B (K x N).

    If `accumulate_c` is True, computes C = A * B + C instead.

    If `transpose_a` is True, treats A as transposed (i.e., K x M) and computes C = A^T * B.
    If `transpose_b` is True, treats B as transposed (i.e., N x K) and computes C = A * B^T.

    If `has_bias` is True, adds a bias term to the result.
    If `has_relu` is True, applies ReLU activation to the result (after bias if any).
    If `truncate_c` is True, truncates the C to A/B data type after accumulation.
    """

    payload_function_name: ClassVar[str] = "payload"
    memory_manager_class: ClassVar[type[MemoryManager]] = GPUMemoryManager

    M: int = 1024
    N: int = 1024
    K: int = 1024
    ab_type: ir.Type | str | None = None
    c_type: ir.Type | str | None = None
    acc_type: ir.Type | str | None = None
    transpose_a: bool = False
    transpose_b: bool = False
    has_bias: bool = False
    has_relu: bool = False
    accumulate_c: bool = True
    truncate_c: bool = False
    _input_arrays_cache: dict[bool, list[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if isinstance(self.ab_type, str):
            self.ab_type = get_mlir_elem_type(self.ab_type)
        if isinstance(self.c_type, str):
            self.c_type = get_mlir_elem_type(self.c_type)
        if isinstance(self.acc_type, str):
            self.acc_type = get_mlir_elem_type(self.acc_type)
        if self.ab_type is None:
            self.ab_type = ir.F16Type.get()
        if self.acc_type is None:
            self.acc_type = ir.F32Type.get()
        if self.c_type is None:
            self.c_type = self.ab_type if self.truncate_c else self.acc_type
        assert isinstance(self.ab_type, ir.F16Type), (
            "Only f16 type is supported for A and B"
        )
        assert isinstance(self.acc_type, ir.F32Type), "Only f32 type is supported for C"
        if self.truncate_c:
            assert self.c_type == self.ab_type, (
                "C type must match A/B type when truncating"
            )
        self.ab_dtype = mlir_to_numpy_dtype(self.ab_type)
        self.c_dtype = mlir_to_numpy_dtype(self.c_type)
        self.a_shape = (self.M, self.K) if not self.transpose_a else (self.K, self.M)
        self.b_shape = (self.K, self.N) if not self.transpose_b else (self.N, self.K)
        self.c_shape = (self.M, self.N)
        self.bias_shape = (self.N,)

    def get_input_arrays(self, init_int: bool = False) -> list[np.ndarray]:
        """Generate initial values on host with numpy."""

        # Cache the generated arrays to avoid regenerating them for every run.
        cached = self._input_arrays_cache.get(init_int)
        if cached is not None:
            return cached

        def gen_random(shape, dtype):
            if init_int:
                # Use integer values to avoid f16/f32 floating point
                # discrepancies in the correctness check.
                a = np.random.randint(-3, 4, shape)
            else:
                # Use float values for benchmarking to get reliable performance
                # measurements.
                a = np.random.rand(*shape) - 0.5
            return a.astype(dtype)

        np.random.seed(2)
        A = gen_random(self.a_shape, self.ab_dtype)
        B = gen_random(self.b_shape, self.ab_dtype)
        C = gen_random(self.c_shape, self.c_dtype)
        if self.has_bias:
            bias = gen_random(self.bias_shape, self.c_dtype)
            arrays = [C, A, B, bias]
        else:
            arrays = [C, A, B]
        self._input_arrays_cache[init_int] = arrays
        return arrays

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes_ab = np.dtype(self.ab_dtype).itemsize
        nbytes_c = np.dtype(self.c_dtype).itemsize
        return matmul_complexity(
            self.M,
            self.N,
            self.K,
            self.has_bias,
            self.has_relu,
            self.accumulate_c,
            nbytes_ab,
            nbytes_c,
        )

    def payload_module(self) -> ir.Module:
        mod = generate_gpu_matmul_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            K=self.K,
            ab_type=self.ab_type,
            c_type=self.acc_type,
            result_type=self.c_type if not self.truncate_c else self.ab_type,
            transpose_a=self.transpose_a,
            transpose_b=self.transpose_b,
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=self.accumulate_c,
        )
        ranks_and_types = list(set(((2, self.ab_type), (2, self.c_type))))
        if self.has_bias:
            ranks_and_types.append((1, self.c_type))
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        assert parameters is not None, "Schedule parameters must be provided"
        schedules = []
        schedules.append(Runner.get_bench_wrapper_schedule(self.payload_function_name))

        schedules.append(
            mlp_schedule(
                stop_at_stage=stop_at_stage,
                params=[parameters],
            )
        )

        if stop_at_stage and stop_at_stage != "final":
            return schedules

        schedules.append(xegpu_to_binary())

        return schedules

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def check_results(
    mmul: XeGPUMatMul,
    host_inputs: list[np.ndarray],
    host_solution: np.ndarray,
    verbose: int = 0,
) -> bool:
    """
    Check correctness of the result.
    """
    # Compute reference solution on host.
    C, A, B = host_inputs[:3]
    bias = host_inputs[3] if mmul.has_bias else None

    if mmul.transpose_a:
        A = A.T
    if mmul.transpose_b:
        B = B.T

    # use float32 data type for efficiency
    f32 = np.float32
    D_ref = A.astype(f32) @ B.astype(f32)
    if mmul.accumulate_c:
        D_ref += C.astype(f32)
    if mmul.has_bias:
        D_ref += bias.astype(f32)
    if mmul.has_relu:
        D_ref = np.maximum(D_ref, 0)

    D_host = host_solution.astype(np.float32)
    if verbose > 1:
        print("Reference solution:")
        print(D_ref)
        print("Computed solution:")
        print(D_host)
    success = np.allclose(D_host, D_ref)

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED Result mismatch!")

    return success


def cli_parser(description):
    """CLI argument parser for args shared with autotuner."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all-knobs", action="store_true", help="Use knobs for all schedule parameters"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN).",
    )
    parser.add_argument(
        "--transpose-a",
        action="store_true",
        help="Transpose matrix A (i.e., A is KxM) before multiplication.",
    )
    parser.add_argument(
        "--transpose-b",
        action="store_true",
        help="Transpose matrix B (i.e., B is NxK) before multiplication.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Add bias after the matrix multiplication.",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Add relu op after the matrix multiplication (and bias if any).",
    )
    parser.add_argument(
        "--no-accumulate-c",
        action="store_true",
        help="Compute plain matrix-multiply C=A*B instead of matrix-multiply-accumulate C+=A*B.",
    )
    parser.add_argument(
        "--truncate-c",
        action="store_true",
        help="Truncate C to A,B data type after accumulation (e.g., float32 to float16).",
    )
    return parser


def parse_cli_args(description):
    parser = cli_parser(description=description)
    parser.add_argument(
        "--wg-tile",
        type=int,
        nargs=2,
        help="Workgroup tile size M,N.",
    )
    parser.add_argument(
        "--sg-tile",
        type=int,
        nargs=2,
        help="Subgroup tile size M,N.",
    )
    parser.add_argument(
        "--k-tile",
        type=int,
        help="Inner reduction dimension tile size K.",
    )
    parser.add_argument(
        "--load-tile-a",
        type=int,
        nargs=2,
        help="Tile size for loading A matrix for DPAS op.",
    )
    parser.add_argument(
        "--load-tile-b",
        type=int,
        nargs=2,
        help="Tile size for loading B matrix for DPAS op.",
    )
    parser.add_argument(
        "--prefetch-tile-a",
        type=int,
        nargs=2,
        help="Tile size for cooperative prefetching of subgroup A matrix",
    )
    parser.add_argument(
        "--prefetch-tile-b",
        type=int,
        nargs=2,
        help="Tile size for cooperative prefetching of subgroup B matrix",
    )
    parser.add_argument(
        "--prefetch-a-nb",
        type=int,
        help="Number of initial prefetches for A matrix.",
    )
    parser.add_argument(
        "--prefetch-b-nb",
        type=int,
        help="Number of initial prefetches for B matrix.",
    )
    parser.add_argument(
        "--init-int",
        action="store_true",
        help="Initialize arrays with integers in [-3, 3] to make result checking more reliable.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the matrix multiplication.",
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
        "--json",
        help="Read problem sizes and tile parameters from a JSON file.",
    )
    parser.add_argument(
        "--target",
        choices=["B70", "B50"],
        help="Target GPU device, e.g., B70.",
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
    description = """XeGPU matrix multiplication example with tunable parameters.

If run without arguments, executes a M=N=K=4096 matrix-multiply-accumulate
kernel without bias or relu and default tile sizes. The problem size and tile
sizes can be overridden by providing a JSON file or using the CLI arguments.
CLI arguments take precedence over everything else. Bias and relu can only be
enabled via CLI arguments.
"""
    args = parse_cli_args(description=description)

    # Problem size
    m, n, k = args.sizes if args.sizes else (4096, 4096, 4096)
    transpose_a = args.transpose_a
    transpose_b = args.transpose_b

    # Set required parameters
    params = {
        "m": m,
        "n": n,
        "k": k,
        "transpose_a": transpose_a,
        "transpose_b": transpose_b,
    }

    # Collect parameters from CLI arguments
    cli_params = {}
    if args.wg_tile:
        cli_params["wg_m"], cli_params["wg_n"] = args.wg_tile
    if args.sg_tile:
        cli_params["sg_m"], cli_params["sg_n"] = args.sg_tile
    if args.k_tile:
        cli_params["k_tile"] = args.k_tile
    if args.load_tile_a:
        cli_params["load_a_m"], cli_params["load_a_k"] = args.load_tile_a
    if args.load_tile_b:
        cli_params["load_b_k"], cli_params["load_b_n"] = args.load_tile_b
    if args.prefetch_tile_a:
        cli_params["prefetch_a_m"], cli_params["prefetch_a_k"] = args.prefetch_tile_a
    if args.prefetch_tile_b:
        cli_params["prefetch_b_k"], cli_params["prefetch_b_n"] = args.prefetch_tile_b
    if args.prefetch_a_nb is not None:
        cli_params["prefetch_a_nb"] = args.prefetch_a_nb
    if args.prefetch_b_nb is not None:
        cli_params["prefetch_b_nb"] = args.prefetch_b_nb

    # By default the tile size parameters are left undefined
    if args.json:
        # Override parameters with values from JSON file if provided
        with open(args.json, "r") as f:
            json_params = json.load(f)
        params.update(json_params)
        # Override with CLI params
        params.update(cli_params)
    elif cli_params:
        # Get default parameters from selector
        param_selector = XeGPUParameterSelector(device=args.target)
        def_params = param_selector.get_parameters((m, n, k), transpose_a, transpose_b)
        params.update(def_params)
        # Override with CLI params
        params.update(cli_params)

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMatMul(
            M=params["m"],
            N=params["n"],
            K=params["k"],
            transpose_a=params["transpose_a"],
            transpose_b=params["transpose_b"],
            has_bias=args.bias,
            has_relu=args.relu,
            accumulate_c=not args.no_accumulate_c,
            truncate_c=args.truncate_c,
        )

        if args.check_result and not args.init_int:
            warnings.warn(
                "Correctness checking with float initialization is not reliable: "
                "it may report a mismatch even when the kernel is correct, due to "
                "floating-point rounding differences. Consider using --init-int.",
                RuntimeWarning,
            )

        if args.dump_kernel or args.dump_schedule:
            if args.dump_kernel:
                pipeline = TransformDriver(
                    wload.schedule_modules(
                        stop_at_stage=args.dump_kernel, parameters=params
                    )
                )
                payload = pipeline.apply(wload.payload_module())
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
            host_inputs = wload.get_input_arrays(args.init_int)
            if args.check_result:
                # Setup callback function to copy result from device to host.
                D_host_copy = np.zeros(wload.c_shape, dtype=wload.c_dtype)
                argument_access_callback = Runner.get_gpu_argument_access_callback(
                    D_host_copy, arg_index=0
                )

                runner.execute(
                    host_input_buffers=host_inputs,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )
                success = check_results(
                    wload,
                    host_inputs,
                    D_host_copy,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")

            times = runner.benchmark(
                host_input_buffers=host_inputs,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            ab_type = str(wload.ab_type)
            c_type = str(wload.c_type)
            print(
                f"sizes={list2str([params['m'], params['n'], params['k']])} "
                f"ta={int(params['transpose_a'])} "
                f"tb={int(params['transpose_b'])} "
                f"dt={ab_type},{c_type} "
                f"wg={list2str([params['wg_m'], params['wg_n']])} "
                f"sg={list2str([params['sg_m'], params['sg_n']])} "
                f"k={params['k_tile']} "
                f"ld-a={list2str([params['load_a_m'], params['load_a_k']])} "
                f"ld-b={list2str([params['load_b_k'], params['load_b_n']])} "
                f"pf-a={list2str([params['prefetch_a_m'], params['prefetch_a_k']])} "
                f"pf-b={list2str([params['prefetch_b_k'], params['prefetch_b_n']])} "
                f"pf-a-nb={params['prefetch_a_nb']} "
                f"pf-b-nb={params['prefetch_b_nb']} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f}"
            )

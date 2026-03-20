# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --relu | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias --relu | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --no-accumulate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --bias --relu --no-accumulate-c | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU matrix multiplication example.
"""

import argparse
import ctypes
import json
from typing import Optional
from functools import cached_property

import numpy as np
from mlir import ir
from mlir.execution_engine import ExecutionEngine

from lighthouse import dialects as lh_dialects
from lighthouse.workload import benchmark, get_bench_wrapper_schedule
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_ctype
from lighthouse.schedule.xegpu.mlp_schedule import get_schedule_module
from lighthouse.ingress.mlir_gen import (
    generate_gpu_matmul_payload,
    get_mlir_elem_type,
)

from xegpu_workload import XeGPUWorkload, matmul_complexity
import parameter_selector


class XeGPUMatMul(XeGPUWorkload):
    """
    Matrix multiplication workload on XeGPU.

    Computes C = A * B for input matrices A (M x K) and B (K x N).

    Optionally adds a ReLU operation on the result C.
    Optionally adds a bias term to C (not implemented yet).
    """

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        ab_type: str = "f16",
        c_type: str = "f32",
        has_bias: bool = False,
        has_relu: bool = False,
        accumulate_c: bool = True,
    ):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.a_shape = (M, K)
        self.b_shape = (K, N)
        self.c_shape = (M, N)
        self.bias_shape = (N,)
        assert ab_type == "f16", "Only f16 type is supported for A and B"
        assert c_type == "f32", "Only f32 type is supported for C"
        self.ab_type = ab_type
        self.c_type = c_type
        type_str_to_numpy = {
            "f16": np.float16,
            "f32": np.float32,
        }
        self.ab_dtype = type_str_to_numpy[ab_type]
        self.c_dtype = type_str_to_numpy[c_type]
        self.has_bias = has_bias
        self.has_relu = has_relu
        self.accumulate_c = accumulate_c

    @cached_property
    def _initial_host_arrays(self) -> list[np.ndarray]:
        """Generate initial values on host with numpy."""

        # use integer values to avoid f16/f32 floating point discrepancies
        def gen_random(shape, dtype):
            # generate values in range [-3, 3]
            a = np.random.randint(-3, 4, shape)
            return a.astype(dtype)

        np.random.seed(2)
        A = gen_random(self.a_shape, self.ab_dtype)
        B = gen_random(self.b_shape, self.ab_dtype)
        C = gen_random(self.c_shape, self.c_dtype)
        bias = None
        if self.has_bias:
            bias = gen_random(self.bias_shape, self.c_dtype)
        return C, A, B, bias

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        """Compute reference solution on host with numpy."""
        C, A, B, bias = self._initial_host_arrays
        # use float32 data type for efficiency
        f32 = np.float32
        C_ref = A.astype(f32) @ B.astype(f32)
        if self.accumulate_c:
            C_ref += C.astype(f32)
        if self.has_bias:
            C_ref += bias.astype(f32)
        if self.has_relu:
            C_ref = np.maximum(C_ref, 0)
        return C_ref

    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        # Allocate device memory for inputs and outputs.
        A_gpu = self._allocate_array("A", self.a_shape, self.ab_type, execution_engine)
        B_gpu = self._allocate_array("B", self.b_shape, self.ab_type, execution_engine)
        C_gpu = self._allocate_array("C", self.c_shape, self.c_type, execution_engine)
        if self.has_bias:
            bias_gpu = self._allocate_array(
                "bias", self.bias_shape, self.c_type, execution_engine
            )

        # Copy initial values to device.
        C_host, A_host, B_host, bias_host = self._initial_host_arrays
        copy_ab, copy_c = ("gpu_copy_2d_" + s for s in (self.ab_type, self.c_type))
        execution_engine.invoke(copy_ab, numpy_to_ctype(A_host), memref_to_ctype(A_gpu))
        execution_engine.invoke(copy_ab, numpy_to_ctype(B_host), memref_to_ctype(B_gpu))
        execution_engine.invoke(copy_c, numpy_to_ctype(C_host), memref_to_ctype(C_gpu))
        if self.has_bias:
            copy_bias = "gpu_copy_1d_" + self.c_type
            execution_engine.invoke(
                copy_bias, numpy_to_ctype(bias_host), memref_to_ctype(bias_gpu)
            )

        # Return memrefs for the payload function.
        if self.has_bias:
            return [C_gpu, A_gpu, B_gpu, bias_gpu]
        return [C_gpu, A_gpu, B_gpu]

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        # Copy result from device to host.
        C_gpu = self.gpu_memrefs[("C", self.c_type)]
        C_host_copy = np.zeros((self.M, self.N), dtype=self.c_dtype)
        execution_engine.invoke(
            "gpu_copy_2d_" + self.c_type,
            memref_to_ctype(C_gpu),
            numpy_to_ctype(C_host_copy),
        )

        C_host_ref = self._reference_solution
        C_host = C_host_copy.astype(np.float32)
        if verbose > 1:
            print("Reference solution:")
            print(C_host_ref)
            print("Computed solution:")
            print(C_host)
        success = np.allclose(C_host, C_host_ref)

        if verbose:
            if success:
                print("PASSED")
            else:
                print("FAILED Result mismatch!")
        return success

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
            ab_type=get_mlir_elem_type(self.ab_type),
            c_type=get_mlir_elem_type(self.c_type),
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=self.accumulate_c,
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        assert parameters is not None, "Schedule parameters must be provided"
        return [
            get_bench_wrapper_schedule(self),
            get_schedule_module(
                has_bias=self.has_bias,
                has_relu=self.has_relu,
                has_convert_c=False,
                stop_at_stage=stop_at_stage,
                params=[parameters],
            ),
        ]

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


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
        "--prefetch-nb",
        type=int,
        help="Number of initial prefetches.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the matrix multiplication.",
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

    ab_type = "f16"
    c_type = "f32"

    # Problem size
    m, n, k = args.sizes if args.sizes else (4096, 4096, 4096)
    # Get default parameters from the database
    try:
        params = parameter_selector.get_matmul_parameters(m, n, k)
    except ValueError:
        # Initialize with a stub and assume the rest will be populated
        params = {
            "m": m,
            "n": n,
            "k": k,
            "wg_m": None,
            "wg_n": None,
            "sg_m": None,
            "sg_n": None,
            "k_tile": None,
            "load_a_m": None,
            "load_a_k": None,
            "load_b_k": None,
            "load_b_n": None,
            "prefetch_a_m": None,
            "prefetch_a_k": None,
            "prefetch_b_k": None,
            "prefetch_b_n": None,
            "prefetch_nb": None,
        }
    if args.json:
        # Override parameters with values from JSON file if provided
        with open(args.json, "r") as f:
            json_params = json.load(f)
        params.update(json_params)

    # Override parameters with CLI args if provided
    if args.wg_tile:
        params["wg_m"], params["wg_n"] = args.wg_tile
    if args.sg_tile:
        params["sg_m"], params["sg_n"] = args.sg_tile
    if args.k_tile:
        params["k_tile"] = args.k_tile
    if args.load_tile_a:
        params["load_a_m"], params["load_a_k"] = args.load_tile_a
    if args.load_tile_b:
        params["load_b_k"], params["load_b_n"] = args.load_tile_b
    if args.prefetch_tile_a:
        params["prefetch_a_m"], params["prefetch_a_k"] = args.prefetch_tile_a
    if args.prefetch_tile_b:
        params["prefetch_b_k"], params["prefetch_b_n"] = args.prefetch_tile_b
    if args.prefetch_nb is not None:
        params["prefetch_nb"] = args.prefetch_nb

    for k, v in params.items():
        if v is None:
            raise ValueError(
                f"Parameter {k} is not set. Please provide it via CLI or JSON file."
            )

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMatMul(
            M=params["m"],
            N=params["n"],
            K=params["k"],
            ab_type=ab_type,
            c_type=c_type,
            has_bias=args.bias,
            has_relu=args.relu,
            accumulate_c=not args.no_accumulate_c,
        )

        if args.dump_kernel or args.dump_schedule:
            wload.lower_payload(
                dump_payload=args.dump_kernel,
                dump_schedule=args.dump_schedule,
                schedule_parameters=params,
            )
        else:
            times = benchmark(
                wload,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
                schedule_parameters=params,
                check_correctness=args.check_result,
                verbose=1,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            print(
                f"sizes={list2str([params['m'], params['n'], params['k']])} "
                f"dt={ab_type},{c_type} "
                f"wg-tile={list2str([params['wg_m'], params['wg_n']])} "
                f"sg-tile={list2str([params['sg_m'], params['sg_n']])} "
                f"k-tile={params['k_tile']} "
                f"load-a-tile={list2str([params['load_a_m'], params['load_a_k']])} "
                f"load-b-tile={list2str([params['load_b_k'], params['load_b_n']])} "
                f"pf-a-tile={list2str([params['prefetch_a_m'], params['prefetch_a_k']])} "
                f"pf-b-tile={list2str([params['prefetch_b_k'], params['prefetch_b_n']])} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f}"
            )

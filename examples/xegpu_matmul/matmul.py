# RUN: %PYTHON %s --dump-payload=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU matrix multiplication benchmark.
"""

import argparse
import ctypes
from typing import Optional
from contextlib import contextmanager
from functools import cached_property

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import (
    get_ranked_memref_descriptor,
    make_nd_memref_descriptor,
    as_ctype,
)
from mlir.execution_engine import ExecutionEngine

from lighthouse.workload import Workload, benchmark
from lighthouse.utils.memref import get_packed_arg, to_ctype as memref_to_ctype

# Import from sibling files:
from schedule import get_schedule_module
from payload import generate_matmul_payload


def numpy_to_ctype(arr: np.ndarray) -> ctypes._Pointer:
    """Convert numpy array to memref and ctypes **void pointer."""
    return memref_to_ctype(get_ranked_memref_descriptor(arr))


class XeGPUMatMul(Workload):
    """
    Matrix multiplication workload on XeGPU.

    Computes C = A * B for input matrices A (M x K) and B (K x N).

    Optionally adds a ReLU operation on the result C.
    Optionally adds a bias term to C (not implemented yet).
    """

    payload_function_name = "payload"

    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        ab_type: str = "f16",
        c_type: str = "f32",
        has_bias: bool = False,
        has_relu: bool = False,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.a_shape = (M, K)
        self.b_shape = (K, N)
        self.c_shape = (M, N)
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
        if has_bias:
            raise NotImplementedError("Bias is not implemented yet")
        # cache allocated memrefs
        self.gpu_memrefs = {}

    def _allocate_array(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype_str: str,
        execution_engine: ExecutionEngine,
    ) -> ctypes.Structure:
        key = (name, dtype_str)
        if key in self.gpu_memrefs:
            return self.gpu_memrefs[key]
        dtype = {
            "f16": np.float16,
            "f32": np.float32,
        }[dtype_str]
        alloc_func = execution_engine.lookup("gpu_alloc_" + dtype_str)
        mref = make_nd_memref_descriptor(len(shape), as_ctype(dtype))()
        ptr_mref = ctypes.pointer(ctypes.pointer(mref))
        ptr_dims = [ctypes.pointer(ctypes.c_int32(d)) for d in shape]
        alloc_func(get_packed_arg([ptr_mref] + ptr_dims))
        self.gpu_memrefs[key] = mref
        return mref

    def _deallocate_all(self, execution_engine: ExecutionEngine):
        for (_, dtype_str), mref in self.gpu_memrefs.items():
            dealloc_func = execution_engine.lookup("gpu_dealloc_" + dtype_str)
            ptr_mref = ctypes.pointer(ctypes.pointer(mref))
            dealloc_func(get_packed_arg([ptr_mref]))
        self.gpu_memrefs = {}

    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        try:
            yield self._get_input_arrays(execution_engine)
        finally:
            self._deallocate_all(execution_engine)

    @cached_property
    def _initial_host_arrays(self) -> list[np.ndarray]:
        """Generate initial values on host with numpy."""

        # use integer values to avoid f16/f32 floating point discrepancies
        def gen_random(shape, dtype):
            # generate values in range [-3, 3]
            a = np.round(6 * np.random.random_sample(shape)) - 3
            return a.astype(dtype)

        np.random.seed(2)
        A = gen_random((self.M, self.K), self.ab_dtype)
        B = gen_random((self.K, self.N), self.ab_dtype)
        C = gen_random((self.M, self.N), self.c_dtype)
        return A, B, C

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        """Compute reference solution on host with numpy."""
        A, B, C = self._initial_host_arrays
        # use float32 data type for efficiency
        f32 = np.float32
        C_ref = A.astype(f32) @ B.astype(f32) + C.astype(f32)
        if self.has_relu:
            C_ref = np.maximum(C_ref, 0)
        if self.has_bias:
            raise NotImplementedError("Bias verification not implemented")
        return C_ref

    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        A_gpu = self._allocate_array("A", self.a_shape, self.ab_type, execution_engine)
        B_gpu = self._allocate_array("B", self.b_shape, self.ab_type, execution_engine)
        C_gpu = self._allocate_array("C", self.c_shape, self.c_type, execution_engine)

        A_host, B_host, C_host = self._initial_host_arrays
        # copy initial values to device
        copy_func_ab = execution_engine.lookup("gpu_copy_" + self.ab_type)
        copy_func_c = execution_engine.lookup("gpu_copy_" + self.c_type)
        copy_func_ab(get_packed_arg([numpy_to_ctype(A_host), memref_to_ctype(A_gpu)]))
        copy_func_ab(get_packed_arg([numpy_to_ctype(B_host), memref_to_ctype(B_gpu)]))
        copy_func_c(get_packed_arg([numpy_to_ctype(C_host), memref_to_ctype(C_gpu)]))

        # return memrefs for the payload function
        return [A_gpu, B_gpu, C_gpu]

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        # copy result from device to host
        C_gpu = self.gpu_memrefs[("C", self.c_type)]
        C_host_copy = np.zeros((self.M, self.N), dtype=self.c_dtype)
        copy_func = execution_engine.lookup("gpu_copy_" + self.c_type)
        copy_func(get_packed_arg([memref_to_ctype(C_gpu), numpy_to_ctype(C_host_copy)]))

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
        M, N, K = self.M, self.N, self.K
        flop_count = 2 * M * N * K
        if self.has_bias:
            flop_count += M * N
        if self.has_relu:
            flop_count += M * N
        nbytes_ab = np.dtype(self.ab_dtype).itemsize
        nbytes_c = np.dtype(self.c_dtype).itemsize
        memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
        memory_writes = M * N * nbytes_c  # write C
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        mod = generate_matmul_payload(
            func_name=self.payload_function_name,
            M=self.M,
            N=self.N,
            K=self.K,
            ab_type_str=self.ab_type,
            c_type_str=self.c_type,
            has_bias=self.has_bias,
            has_relu=self.has_relu,
        )
        return mod

    def schedule_module(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> ir.Module:
        return get_schedule_module(
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            stop_at_stage=stop_at_stage,
            params=parameters,
        )

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Matrix Multiplication using MLIR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN).",
    )
    parser.add_argument(
        "--wg-tile",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Workgroup tile size M,N.",
    )
    parser.add_argument(
        "--sg-tile",
        type=int,
        nargs=2,
        default=[32, 32],
        help="Subgroup tile size M,N.",
    )
    parser.add_argument(
        "--k-tile",
        type=int,
        default=32,
        help="Inner reduction dimension tile size K.",
    )
    parser.add_argument(
        "--load-tile-a",
        type=int,
        nargs=2,
        default=[32, 16],
        help="Tile size for loading A matrix for DPAS op.",
    )
    parser.add_argument(
        "--load-tile-b",
        type=int,
        nargs=2,
        default=[32, 16],
        help="Tile size for loading B matrix for DPAS op.",
    )
    parser.add_argument(
        "--prefetch-tile-a",
        type=int,
        nargs=2,
        default=[8, 32],
        help="Tile size for cooperative prefetching of subgroup A matrix",
    )
    parser.add_argument(
        "--prefetch-tile-b",
        type=int,
        nargs=2,
        default=[8, 16],
        help="Tile size for cooperative prefetching of subgroup B matrix",
    )
    parser.add_argument(
        "--nb-prefetch",
        type=int,
        default=1,
        help="Number of initial prefetches.",
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
        "--relu",
        action="store_true",
        help="Add relu op after the matrix multiplication (and bias if any).",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the matrix multiplication.",
    )
    parser.add_argument(
        "--dump-payload",
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
        help="Dump payload IR at different stages of lowering.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    parser.add_argument(
        "--non-det",
        action="store_true",
        help="Generate schedule with knob values left non-determined.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli()

    params = {
        "wg_d0": args.wg_tile[0],
        "wg_d1": args.wg_tile[1],
        "sg_d0": args.sg_tile[0],
        "sg_d1": args.sg_tile[1],
        "k_tile": args.k_tile,
        "load_a_d0": args.load_tile_a[0],
        "load_a_d1": args.load_tile_a[1],
        "load_b_d0": args.load_tile_b[0],
        "load_b_d1": args.load_tile_b[1],
        "prefetch_a_d0": args.prefetch_tile_a[0],
        "prefetch_a_d1": args.prefetch_tile_a[1],
        "prefetch_b_d0": args.prefetch_tile_b[0],
        "prefetch_b_d1": args.prefetch_tile_b[1],
        "nb_prefetch": args.nb_prefetch,
    }
    if args.non_det:
        params = {}

    M, N, K = args.sizes
    ab_type = "f16"
    c_type = "f32"

    with ir.Context(), ir.Location.unknown():
        wload = XeGPUMatMul(
            M=M,
            N=N,
            K=K,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=False,
            has_relu=args.relu,
        )

        if args.dump_schedule:
            schedule_module = wload.schedule_module(
                stop_at_stage=args.dump_payload, parameters=params
            )
            print(schedule_module)
        elif args.dump_payload:
            wload.lower_payload(
                dump_payload=args.dump_payload,
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

            parts = [
                f"sizes={list2str(args.sizes)}",
                f"dt={ab_type},{c_type}",
                f"wg-tile={list2str(args.wg_tile)}",
                f"sg-tile={list2str(args.sg_tile)}",
                f"k-tile={args.k_tile}",
                f"load-a-tile={list2str(args.load_tile_a)}",
                f"load-b-tile={list2str(args.load_tile_b)}",
                f"pf-a-tile={list2str(args.prefetch_tile_a)}",
                f"pf-b-tile={list2str(args.prefetch_tile_b)}",
                f"time(us): {elapsed:.2f}",
                f"GFLOPS: {gflops:.2f}",
            ]
            print(" ".join(parts))

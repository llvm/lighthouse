# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU MLP benchmark.
"""

import argparse
import ctypes
from typing import Optional
from contextlib import contextmanager
from functools import cached_property
import warnings

import numpy as np
from mlir import ir
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    as_ctype,
)
from mlir.execution_engine import ExecutionEngine

from lighthouse.workload import Workload, benchmark
from lighthouse.utils.memref import get_packed_arg, to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_ctype
from lighthouse.schedule.xegpu.mlp_schedule import get_schedule_module
from lighthouse.ingress.gpu import generate_mlp_payload

import parameter_selector


class XeGPUMLP(Workload):
    """
    Multi-layer perceptron (MLP) workload on XeGPU.

    Optionally adds a ReLU operation after each layer.
    Optionally adds a bias term in each layer (not implemented yet).
    """

    payload_function_name = "payload"

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: Optional[list[int]] = None,
        ab_type: str = "f16",
        c_type: str = "f32",
        has_bias: bool = False,
        has_relu: bool = False,
        accumulate_c: bool = False,
        identity_weights: bool = False,
    ):
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes or []
        self.input_shape = (self.batch_size, self.input_size)
        self.output_shape = (self.batch_size, self.output_size)
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        self.weight_shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.matmul_layers = [(self.batch_size, o, i) for i, o in self.weight_shapes]
        self.identity_weights = identity_weights

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
        if has_bias:
            raise NotImplementedError("Bias is not implemented yet")

        if len(self.matmul_layers) == 1 and self.has_relu:
            warnings.warn("Using ReLU on a single layer model has no effect.")

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

        def gen_identity(shape, dtype):
            # identity matrix, if cols > rows wrap to fill all columns
            a = np.zeros(shape, dtype=dtype)
            np.fill_diagonal(a, 1)
            if shape[1] > shape[0]:
                second_block = a[:, shape[0] :]
                np.fill_diagonal(second_block, 1)
            return a

        np.random.seed(2)
        input_array = gen_random(self.input_shape, self.ab_dtype)
        output_array = np.zeros(self.output_shape, self.ab_dtype)
        weights = []
        for i, o in self.weight_shapes:
            if self.identity_weights:
                W = gen_identity((i, o), self.ab_dtype)
            else:
                W = gen_random((i, o), self.ab_dtype)
            weights.append(W)

        if self.has_bias:
            raise NotImplementedError("Bias initialization not implemented")

        return input_array, output_array, *weights

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        """Compute reference solution on host with numpy."""
        # NOTE for large problems the solution can overflow float16 range
        host_arrays = self._initial_host_arrays
        # use float32 data type for efficiency
        host_arrays = [arr.astype(np.float32) for arr in host_arrays]
        input_array = host_arrays[0]
        output_array = host_arrays[1]
        weights = host_arrays[2:]

        a_array = input_array
        for i, W in enumerate(weights):
            C_ref = a_array @ W
            if self.has_relu and i < len(weights) - 1:
                C_ref = np.maximum(C_ref, 0)
            if self.has_bias:
                raise NotImplementedError("Bias verification not implemented")
            a_array = C_ref.astype(self.ab_dtype).astype(np.float32)

        C_ref += output_array
        return C_ref.astype(self.ab_dtype)

    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        if self.has_bias:
            raise NotImplementedError("Bias allocation not implemented yet")

        # allocate arrays on device
        input_gpu = self._allocate_array(
            "input", self.input_shape, self.ab_type, execution_engine
        )
        output_gpu = self._allocate_array(
            "output", self.output_shape, self.ab_type, execution_engine
        )
        gpu_arrays = [input_gpu, output_gpu]
        for i, (in_size, out_size) in enumerate(self.weight_shapes):
            W_gpu = self._allocate_array(
                f"weight_{i}", (in_size, out_size), self.ab_type, execution_engine
            )
            gpu_arrays.append(W_gpu)

        # get initial host arrays
        host_arrays = self._initial_host_arrays
        # copy initial values to device
        copy_func_ab = execution_engine.lookup("gpu_copy_" + self.ab_type)
        for host_arr, gpu_arr in zip(host_arrays, gpu_arrays):
            copy_func_ab(
                get_packed_arg([numpy_to_ctype(host_arr), memref_to_ctype(gpu_arr)])
            )

        # return memrefs for the payload function
        return gpu_arrays

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        # copy result from device to host
        res_gpu = self.gpu_memrefs[("output", self.ab_type)]
        res_host_copy = np.zeros(self.output_shape, dtype=self.ab_dtype)
        copy_func = execution_engine.lookup("gpu_copy_" + self.ab_type)
        copy_func(
            get_packed_arg([memref_to_ctype(res_gpu), numpy_to_ctype(res_host_copy)])
        )

        res_host_ref = self._reference_solution
        res_host = res_host_copy
        if verbose > 1:
            print("Reference solution:")
            print(res_host_ref)
            print("Computed solution:")
            print(res_host)
        success = np.allclose(res_host, res_host_ref)

        if verbose:
            if success:
                print("PASSED")
            else:
                print("FAILED Result mismatch!")
                print(f"Max absolute error: {np.max(np.abs(res_host - res_host_ref))}")
                num_diff = np.sum(np.abs(res_host - res_host_ref) > 1e-3)
                print(f"Number of differing elements: {num_diff}")
        return success

    def get_complexity(self) -> tuple[int, int, int]:
        nbytes_ab = np.dtype(self.ab_dtype).itemsize
        nbytes_c = np.dtype(self.c_dtype).itemsize

        def matmul_complexity(M, N, K, has_bias, has_relu):
            flop_count = 2 * M * N * K
            memory_reads = (M * K + K * N) * nbytes_ab  # read A and B
            memory_writes = M * N * nbytes_c  # write C
            if has_bias:
                flop_count += M * N
                memory_reads += N * nbytes_c  # read bias
            if has_relu:
                flop_count += M * N
            return flop_count, memory_reads, memory_writes

        flop_count = 0
        memory_reads = 0
        memory_writes = 0
        for i, (M, N, K) in enumerate(self.matmul_layers):
            relu = self.has_relu if i < len(self.matmul_layers) - 1 else False
            f, r, w = matmul_complexity(M, N, K, self.has_bias, relu)
            flop_count += f
            memory_reads += r
            memory_writes += w
        return (flop_count, memory_reads, memory_writes)

    def payload_module(self) -> ir.Module:
        mod = generate_mlp_payload(
            func_name=self.payload_function_name,
            batch_size=self.batch_size,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            ab_type_str=self.ab_type,
            c_type_str=self.c_type,
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=self.accumulate_c,
        )
        return mod

    def schedule_module(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> ir.Module:
        return get_schedule_module(
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            skip_final_layer_relu=True,
            stop_at_stage=stop_at_stage,
            nlayers=len(self.matmul_layers),
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
        "-b",
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size M. Input matrix has shape (M x K).",
    )
    parser.add_argument(
        "-i",
        "--input-size",
        type=int,
        default=1024,
        help="Number of input features K. Input matrix has shape (M x K).",
    )
    parser.add_argument(
        "-o",
        "--output-size",
        type=int,
        default=1024,
        help="Number of output features N. Output matrix has shape (M x N).",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        help="Number of features in each hidden layers.",
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
        help="Add ReLU activation function to each layer except the output layer.",
    )
    parser.add_argument(
        "--accumulate-c",
        action="store_true",
        help="Use matrix-multiply-accumulate layers instead of initializing the "
        "accumulator tile with zeros.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the MLP model. If the result overflows to "
        "inf/nan values, use --identity-weights option.",
    )
    parser.add_argument(
        "--identity-weights",
        action="store_true",
        help="Initialize weights as (extended) identity matrix, useful for "
        "correctness test. Can skew performance measurement.",
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
            "xegpu-sg",
            "xegpu-inst",
            "final",
        ],
        help="Dump kernel IR at different stages of lowering.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Dump transform schedule.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli()

    ab_type = "f16"
    c_type = "f32"

    with ir.Context(), ir.Location.unknown():
        wload = XeGPUMLP(
            batch_size=args.batch_size,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_layer_sizes=args.hidden_sizes,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=False,
            has_relu=args.relu,
            accumulate_c=args.accumulate_c,
            identity_weights=args.identity_weights,
        )
        matmuls = wload.matmul_layers
        print(f"MLP with {len(matmuls)} layers")
        for i, (M, N, K) in enumerate(matmuls):
            print(f"  Layer {i}: M={M}, N={N}, K={K}")

        params = parameter_selector.get_matmul_parameters(wload)

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
                verbose=2,
            )
            times *= 1e6  # convert to microseconds
            elapsed = np.mean(times)
            flop_count = wload.get_complexity()[0]
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            hidden_sizes = args.hidden_sizes if args.hidden_sizes else []
            parts = [
                f"b={args.batch_size}",
                f"i={args.input_size}",
                f"o={args.output_size}",
                f"hs={list2str(hidden_sizes)}",
                f"dt={ab_type},{c_type}",
                f"time(us): {elapsed:.2f}",
                f"GFLOPS: {gflops:.2f}",
            ]
            print(" ".join(parts))

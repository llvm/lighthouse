# RUN: %PYTHON %s --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --hidden-sizes 1024 1024 | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --hidden-sizes 1024 1024 --relu | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --hidden-sizes 1024 1024 --bias | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --hidden-sizes 1024 1024 --accumulate-c | FileCheck %s
# RUN: %PYTHON %s --dump-kernel=xegpu-wg --hidden-sizes 1024 1024 --bias --relu --accumulate-c | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""
XeGPU MLP benchmark.

The tiling strategy for each MLP layer is chosen by the parameter selector.
Consequently, only layers whose sizes the parameter selector supports can be
lowered and executed.
"""

import argparse
import ctypes
from typing import Optional
from functools import cached_property
import warnings

import numpy as np
from mlir import ir
from mlir.execution_engine import ExecutionEngine

from lighthouse import dialects as lh_dialects
from lighthouse.workload import benchmark, get_bench_wrapper_schedule
from lighthouse.utils.memref import to_ctype as memref_to_ctype
from lighthouse.utils.numpy import numpy_to_ctype
from lighthouse.schedule.xegpu.mlp_schedule import get_schedule_module
from lighthouse.ingress.mlir_gen import (
    generate_gpu_mlp_payload,
    get_mlir_elem_type,
)

from xegpu_workload import XeGPUWorkload, matmul_complexity
import parameter_selector


class XeGPUMLP(XeGPUWorkload):
    """
    Multi-layer perceptron (MLP) workload on XeGPU.

    Optionally adds a ReLU operation after each layer.
    Optionally adds a bias term in each layer (not implemented yet).
    """

    def __init__(
        self,
        batch_size: int,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: Optional[list[int]] = None,
        ab_type: str = "f16",
        acc_type: str = "f32",
        has_bias: bool = False,
        has_relu: bool = False,
        accumulate_c: bool = False,
        identity_weights: bool = False,
    ):
        super().__init__()
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
        self.bias_shapes = [(o,) for o in layer_sizes[1:]] if has_bias else []

        assert ab_type == "f16", "Only f16 type is supported for A and B"
        assert acc_type == "f32", "Only f32 type is supported for accumulator"
        self.ab_type = ab_type
        self.acc_type = acc_type
        type_str_to_numpy = {
            "f16": np.float16,
            "f32": np.float32,
        }
        self.ab_dtype = type_str_to_numpy[ab_type]
        self.has_bias = has_bias
        self.has_relu = has_relu
        self.accumulate_c = accumulate_c

        if len(self.matmul_layers) == 1 and self.has_relu:
            warnings.warn("Using ReLU on a single layer model has no effect.")

    @cached_property
    def _initial_host_arrays(self) -> list[np.ndarray]:
        """Generate initial values on host with numpy."""

        # use integer values to avoid f16/f32 floating point discrepancies
        def gen_random(shape, dtype):
            # generate values in range [0, 1)
            return np.random.rand(*shape).astype(dtype)

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

        biases = []
        if self.has_bias:
            for o in self.bias_shapes:
                b = gen_random(o, self.ab_dtype)
                biases.append(b)

        return output_array, input_array, weights, biases

    @cached_property
    def _reference_solution(self) -> np.ndarray:
        """Compute reference solution on host with numpy."""
        # NOTE for large problems the solution can overflow float16 range
        output_array, input_array, weights, biases = self._initial_host_arrays
        # use float32 data type for efficiency
        output_array = output_array.astype(np.float32)
        input_array = input_array.astype(np.float32)
        weights = [w.astype(np.float32) for w in weights]
        biases = [b.astype(np.float32) for b in biases]

        a_array = input_array
        for i, W in enumerate(weights):
            C_ref = a_array @ W
            if self.has_bias:
                C_ref += biases[i]
            if self.has_relu and i < len(weights) - 1:
                C_ref = np.maximum(C_ref, 0)
            a_array = C_ref.astype(self.ab_dtype).astype(np.float32)

        C_ref += output_array
        return C_ref.astype(self.ab_dtype)

    def _get_input_arrays(
        self, execution_engine: ExecutionEngine
    ) -> list[ctypes.Structure]:
        # Allocate device memory for inputs and outputs.
        input_gpu = self._allocate_array(
            "input", self.input_shape, self.ab_type, execution_engine
        )
        output_gpu = self._allocate_array(
            "output", self.output_shape, self.ab_type, execution_engine
        )
        gpu_arrays_2d = [output_gpu, input_gpu]
        for i, (in_size, out_size) in enumerate(self.weight_shapes):
            W_gpu = self._allocate_array(
                f"weight_{i}", (in_size, out_size), self.ab_type, execution_engine
            )
            gpu_arrays_2d.append(W_gpu)
        gpu_arrays_1d = []
        if self.has_bias:
            for i, (out_size,) in enumerate(self.bias_shapes):
                b_gpu = self._allocate_array(
                    f"bias_{i}", (out_size,), self.ab_type, execution_engine
                )
                gpu_arrays_1d.append(b_gpu)

        # Copy initial values to device.
        input_arr, output_arr, weights, biases = self._initial_host_arrays
        for host_arr, gpu_arr in zip([input_arr, output_arr] + weights, gpu_arrays_2d):
            execution_engine.invoke(
                "gpu_copy_2d_" + self.ab_type,
                numpy_to_ctype(host_arr),
                memref_to_ctype(gpu_arr),
            )
        if self.has_bias:
            for host_arr, gpu_arr in zip(biases, gpu_arrays_1d):
                execution_engine.invoke(
                    "gpu_copy_1d_" + self.ab_type,
                    numpy_to_ctype(host_arr),
                    memref_to_ctype(gpu_arr),
                )

        # Return memrefs for the payload function.
        return gpu_arrays_2d + gpu_arrays_1d

    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        # copy result from device to host
        res_gpu = self.gpu_memrefs[("output", self.ab_type)]
        res_host_copy = np.zeros(self.output_shape, dtype=self.ab_dtype)
        execution_engine.invoke(
            "gpu_copy_2d_" + self.ab_type,
            memref_to_ctype(res_gpu),
            numpy_to_ctype(res_host_copy),
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

        flop_count = 0
        memory_reads = 0
        memory_writes = 0
        for i, (M, N, K) in enumerate(self.matmul_layers):
            relu = self.has_relu if i < len(self.matmul_layers) - 1 else False
            f, r, w = matmul_complexity(
                M, N, K, self.has_bias, relu, self.accumulate_c, nbytes_ab, nbytes_ab
            )
            flop_count += f
            memory_reads += r
            memory_writes += w
        return flop_count, memory_reads, memory_writes

    def payload_module(self) -> ir.Module:
        mod = generate_gpu_mlp_payload(
            func_name=self.payload_function_name,
            batch_size=self.batch_size,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            ab_type=get_mlir_elem_type(self.ab_type),
            acc_type=get_mlir_elem_type(self.acc_type),
            bias_type=get_mlir_elem_type(self.ab_type),
            result_type=get_mlir_elem_type(self.ab_type),
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=self.accumulate_c,
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        return [
            get_bench_wrapper_schedule(self),
            get_schedule_module(
                has_bias=self.has_bias,
                has_relu=self.has_relu,
                skip_final_layer_relu=True,
                stop_at_stage=stop_at_stage,
                params=parameters,
            ),
        ]

    def shared_libs(self) -> list[str]:
        return ["libmlir_levelzero_runtime.so"]


def parse_cli():
    parser = argparse.ArgumentParser(
        description="XeGPU MLP example with tunable parameters.",
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
        "--bias",
        action="store_true",
        help="Add bias to each layer.",
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
        help="Dump kernel IR at different stages of lowering and exit without "
        "executing the kernel.",
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

    # use identity weights in correctness check
    # this may affect performance metrics
    identity_weights = args.check_result

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMLP(
            batch_size=args.batch_size,
            input_size=args.input_size,
            output_size=args.output_size,
            hidden_layer_sizes=args.hidden_sizes,
            has_bias=args.bias,
            has_relu=args.relu,
            accumulate_c=args.accumulate_c,
            identity_weights=identity_weights,
        )
        matmuls = wload.matmul_layers
        print(f"MLP with {len(matmuls)} layers")
        for i, (M, N, K) in enumerate(matmuls):
            print(f"  Layer {i}: M={M}, N={N}, K={K}")
        ab_type = wload.ab_type
        acc_type = wload.acc_type

        params = parameter_selector.get_parameters_for_layers(matmuls)

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
            print(
                f"b={args.batch_size} "
                f"i={args.input_size} "
                f"o={args.output_size} "
                f"hs={list2str(hidden_sizes)} "
                f"dt={ab_type},{acc_type} "
                f"time(us): {elapsed:.2f} "
                f"GFLOPS: {gflops:.2f}"
            )

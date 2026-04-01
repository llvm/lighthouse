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
from dataclasses import dataclass
from typing import Optional, ClassVar
from functools import cached_property
import warnings

import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution import (
    MemoryManager,
    GPUMemoryManager,
)
from lighthouse.utils.numpy import mlir_to_numpy_dtype
from lighthouse.schedule.xegpu.mlp_schedule import get_schedule_module
from lighthouse.ingress.mlir_gen import (
    generate_gpu_mlp_payload,
    get_mlir_elem_type,
)

import parameter_selector
from matmul import matmul_complexity


def check_correctness(
    initial_host_arrays: list[np.ndarray],
    result: np.ndarray,
    ab_dtype: np.dtype,
    has_bias: bool = False,
    has_relu: bool = False,
    verbose: int = 0,
) -> bool:
    output_array, input_array, *rest = initial_host_arrays
    if has_bias:
        n = len(rest) // 2
        weights = rest[:n]
        biases = rest[n:]
    else:
        weights = rest
        biases = []
    # use float32 data type for efficiency
    output_array = output_array.astype(np.float32)
    input_array = input_array.astype(np.float32)
    weights = [w.astype(np.float32) for w in weights]
    biases = [b.astype(np.float32) for b in biases]

    a_array = input_array
    for i, W in enumerate(weights):
        D_ref = a_array @ W
        if has_bias:
            D_ref += biases[i]
        if has_relu and i < len(weights) - 1:
            D_ref = np.maximum(D_ref, 0)
        a_array = D_ref.astype(ab_dtype).astype(np.float32)

    D_ref = a_array.astype(ab_dtype)
    D = result
    if verbose > 1:
        print("Reference solution:")
        print(D_ref)
        print("Computed solution:")
        print(D)
    success = np.allclose(D, D_ref)

    if verbose:
        if success:
            print("PASSED")
        else:
            print("FAILED Result mismatch!")
            print(f"Max absolute error: {np.max(np.abs(D - D_ref))}")
            num_diff = np.sum(np.abs(D - D_ref) > 1e-3)
            print(f"Number of differing elements: {num_diff}")
    return success


@dataclass
class XeGPUMLP:
    """
    Multi-layer perceptron (MLP) kernel on XeGPU.

    Optionally adds a ReLU operation after each layer.
    Optionally adds a bias term in each layer (not implemented yet).
    """

    payload_function_name: ClassVar[str] = "payload"
    memory_manager_class: ClassVar[type[MemoryManager]] = GPUMemoryManager

    batch_size: int = 1024
    input_size: int = 1024
    output_size: int = 1024
    hidden_layer_sizes: Optional[list[int]] = None
    ab_type: ir.Type | str | None = None
    acc_type: ir.Type | str | None = None
    has_bias: bool = False
    has_relu: bool = False
    accumulate_c: bool = False
    identity_weights: bool = False

    def __post_init__(self):
        if isinstance(self.ab_type, str):
            self.ab_type = get_mlir_elem_type(self.ab_type)
        if isinstance(self.acc_type, str):
            self.acc_type = get_mlir_elem_type(self.acc_type)
        if self.ab_type is None:
            self.ab_type = ir.F16Type.get()
        if self.acc_type is None:
            self.acc_type = ir.F32Type.get()
        assert isinstance(self.ab_type, ir.F16Type), (
            "Only f16 type is supported for A and B"
        )
        assert isinstance(self.acc_type, ir.F32Type), (
            "Only f32 type is supported for accumulator"
        )
        self.ab_dtype = mlir_to_numpy_dtype(self.ab_type)
        self.acc_dtype = mlir_to_numpy_dtype(self.acc_type)

        if self.hidden_layer_sizes is None:
            self.hidden_layer_sizes = []
        self.input_shape = (self.batch_size, self.input_size)
        self.output_shape = (self.batch_size, self.output_size)
        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
        self.weight_shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
        self.matmul_layers = [(self.batch_size, o, i) for i, o in self.weight_shapes]
        self.bias_shapes = [(o,) for o in layer_sizes[1:]] if self.has_bias else []

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

        return output_array, input_array, *weights, *biases

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
            ab_type=self.ab_type,
            acc_type=self.acc_type,
            bias_type=self.ab_type,
            result_type=self.ab_type,
            has_bias=self.has_bias,
            has_relu=self.has_relu,
            accumulate_c=self.accumulate_c,
        )
        ranks_and_types = [(2, self.ab_type), (2, self.acc_type)]
        if self.has_bias:
            ranks_and_types.append((1, self.ab_type))
        self.memory_manager_class.emit_memory_management_funcs(
            mod, ranks_and_types=ranks_and_types
        )
        return mod

    def schedule_modules(
        self, stop_at_stage: Optional[str] = None, parameters: Optional[dict] = None
    ) -> list[ir.Module]:
        return [
            Runner.get_bench_wrapper_schedule(self.payload_function_name),
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
                result_host_copy, argument_access_callback = (
                    Runner.get_gpu_argument_access_callback(
                        wload.output_shape, wload.ab_dtype
                    )
                )

                # Execute kernel once.
                runner.execute(
                    host_input_buffers=wload._initial_host_arrays,
                    payload_function_name=wload.payload_function_name,
                    argument_access_callback=argument_access_callback,
                )

                # Compute reference solution on host.
                success = check_correctness(
                    wload._initial_host_arrays,
                    result_host_copy,
                    wload.ab_dtype,
                    has_bias=wload.has_bias,
                    has_relu=wload.has_relu,
                    verbose=args.verbose,
                )
                if not success:
                    raise ValueError("Result mismatch!")

            times = runner.benchmark(
                host_input_buffers=wload._initial_host_arrays,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
                argument_access_callback=None,
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

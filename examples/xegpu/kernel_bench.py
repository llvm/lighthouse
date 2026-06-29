# RUN: %PYTHON %s -l 1 -b 1 --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s -l 2 -b 9 --dump-kernel=xegpu-wg | FileCheck %s
# RUN: %PYTHON %s -l 3 -b 1 --dump-kernel=xegpu-wg | FileCheck %s
# CHECK: module attributes {gpu.container_module} {
"""
This script executes KernelBench benchmarks using the XEGPU lowering pipeline.

The KernelBench submodule must be checked out prior to running this script:

    git submodule update --init --recursive

The script saves results to a CSV file named `out_kernelbench.csv` in the
current working directory.

Currently only some 2D gemm and element-wise benchmarks are supported on Level
1. On level 2, only 2D gemm+epilogue benchmarks are supported. On level 3,
the MLP benchmarks are supported.

Usage:

Run level 1 benchmark 1
    python xegpu_kernel_bench.py -l 1 -b 1

Run level 1 benchmarks 1 and 2
    python xegpu_kernel_bench.py -l 1 -b 1 2

Run all level 1 benchmarks
    python xegpu_kernel_bench.py -l 1

Run using bfloat16 datatype
    python xegpu_kernel_bench.py -l 1 -b 1 --datatype bf16

Increase verbosity
    python xegpu_kernel_bench.py -l 1 -b 1 -vv

Run a single benchmark in debug mode (no timeout, exceptions raised
immediately)
    python xegpu_kernel_bench.py -l 1 -b 1 -vv --debug

Dump the kernel at a specific stage of the pipeline, does not execute the
benchmark
    python xegpu_kernel_bench.py -l 1 -b 1 --dump-kernel bufferized

See all commandline options
    python xegpu_kernel_bench.py --help
"""

from pathlib import Path
import glob
from functools import partial
import argparse
import warnings

import torch
import torch._dynamo as dynamo
import os
import json
import numpy as np
from mlir import ir

import lighthouse
from lighthouse import ingress as lh_ingress
from lighthouse import dialects as lh_dialects
from lighthouse.utils.mlir import inspect_payload
from lighthouse.execution.runner import Runner
from lighthouse.schedule.xegpu import XeGPUParameterSelector
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import mlp_schedule, elemwise_schedule, xegpu_to_binary
from lighthouse.pipeline.helper import PipelineInterrupt
from lighthouse.ingress.torch import gpu_backend, TargetDialect
from lighthouse.ingress.torch.compile import TorchMemoryManager
from lighthouse.pipeline.driver import make_function_callable
from tune_matmul_costmodel import optimize_payload, dump_configs_json
from tune_utils import run_with_timeout
from csv_logger import CSVLogger


def inspect_kb_payload(module: ir.Module) -> tuple[str, dict]:
    """
    Extract metadata from the payload IR module.
    """
    func_metadata = inspect_payload(module)
    assert len(func_metadata) == 1, (
        "Expected exactly one payload function in the module."
    )
    payload_func_name = list(func_metadata.keys())[0]
    func_metadata = func_metadata[payload_func_name]
    return payload_func_name, func_metadata


def infer_parameters(mod: ir.Module, verbose: int = 0) -> tuple[dict, str, list[dict]]:
    """
    Inspects payload and selects lowering schedule and tile size parameters.

    Returns:
        func_metadata: Payload function metadata dict.
        schedule kind: Name of selected lowering schedule.
        schedule_parameters: List of parameter dicts, one per layer.
    """
    payload_func_name, func_metadata = inspect_kb_payload(mod)
    if verbose > 0:
        print(f"Payload function name: {payload_func_name}")
        print("Inputs:")
        for i, input_type in enumerate(func_metadata["inputs"]):
            print(
                f"  {i}: shape={input_type.shape}, element_type={input_type.element_type}"
            )

    # get tile size parameters for each matmul layer
    layer_metadata = func_metadata["layers"]
    matmuls = layer_metadata["matmul"]
    elemwise = layer_metadata["elemwise"]
    elemtype_bytes = {
        "f16": 2,
        "bf16": 2,
        "f32": 4,
    }
    if len(matmuls) > 0:
        schedule_params = XeGPUParameterSelector().get_parameters_for_layers(matmuls)
        # check that all matmul dims are powers of 2
        for mmul in matmuls:
            for char, dim in zip(["m", "n", "k"], mmul["shape"]):
                assert dim & (dim - 1) == 0, (
                    f"Matmul shape must be a power of 2, got {char}={dim}"
                )

        # compute flops
        total_flops = 0
        read_bytes = 0
        write_bytes = 0
        for mmul in matmuls:
            m, n, k = mmul["shape"]
            total_flops += 2 * m * n * k
            a_shape = (m, k)
            b_shape = (k, n)
            c_shape = (m, n)
            # assuming result is also cast to ab type
            ab_elemtype = mmul["ab_elemtype"]
            ab_bytes = elemtype_bytes[ab_elemtype]
            read_bytes += (np.prod(a_shape) + np.prod(b_shape)) * ab_bytes
            write_bytes += np.prod(c_shape) * ab_bytes

        schedule_kind = "mlp"
    elif len(elemwise) > 0:
        # TODO estimate flops in a reliable way, now assuming 1 flop per element
        shape = elemwise[0]["shape"]
        res_elemtype = elemwise[0]["elemtype"]
        total_flops = np.prod(shape)
        res_bytes = elemtype_bytes[res_elemtype]
        read_bytes = np.prod(shape) * res_bytes
        write_bytes = np.prod(shape) * res_bytes

        # Use fixed tile sizes for now
        layer_params = {
            "wg_m": 128,
            "wg_n": 256,
            "sg_m": 32,
            "sg_n": 32,
            "load_m": 8,
            "load_n": 16,
        }
        # NOTE assume all elemwise layers will be fused to a single layer
        schedule_params = [layer_params]
        schedule_kind = "elemwise"
    else:
        print("Layers:")
        for k, v in layer_metadata.items():
            print(f"  {k}: {v}")
        raise ValueError("Unsupported payload type")
    func_metadata["total_flops"] = total_flops
    func_metadata["read_bytes"] = read_bytes
    func_metadata["write_bytes"] = write_bytes
    return func_metadata, schedule_kind, schedule_params


def compute_perf_metrics(
    times: np.ndarray, total_flops: float, read_bytes: float, write_bytes: float
) -> tuple[float, float, float]:
    """
    Compute average execution time and achieved GFLOPS from a list of execution times.
    Args:
        times: List of execution times in seconds.
        total_flops: Total number of floating point operations.
    Returns:
        elapsed: Average execution time in microseconds.
        gflops: Achieved GFLOPS.
        bandwidth: Achieved memory bandwidth in GB/s.
    """
    elapsed = np.mean(times) * 1e6  # convert to microseconds
    gflops = total_flops / (elapsed * 1e-6) / 1e9
    bandwidth = (read_bytes + write_bytes) / (elapsed * 1e-6) / 1e9
    return elapsed, gflops, bandwidth


def copy_module(module: ir.Module) -> ir.Module:
    """
    Create a copy of the payload IR module.

    Required in tuning because lowering transforms the payload module in place.
    """
    copied_module = ir.Module.parse(str(module), context=module.context)
    return copied_module


def tune_matmul_layer(
    gemm_specs: dict,
    mod: ir.Module,
    payload_func_name: str,
    torch_all_inputs: list[torch.Tensor],
    total_flops: float,
    read_bytes: float,
    write_bytes: float,
) -> list[tuple[float, dict]]:
    """
    Optimize the given GEMM kernel with cost model tuner.

    Returns:
        executed_configs: list of (gflops, params) tuples, sorted in descending order of performance.
    """

    def compile_and_eval(nruns: int, nwarmup: int, **kwparams) -> ir.Module:
        """Evaluate kernel with given parameters and return the metrics."""
        final_mod = lower_to_llvm(
            copy_module(mod),
            schedule_kind="mlp",
            schedule_params=[kwparams],
            stop_at_stage=None,
            benchmark=True,
            payload_func_name=payload_func_name,
        )
        runner = Runner(
            final_mod,
            mem_manager_cls=TorchMemoryManager,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )
        # result buffers
        times = runner.benchmark(
            host_input_buffers=torch_all_inputs,
            nwarmup=nwarmup,
            nruns=nruns,
        )
        time, gflops, _ = compute_perf_metrics(
            times, total_flops, read_bytes, write_bytes
        )
        return time, gflops

    executed_configs, _, _ = optimize_payload(
        compile_and_eval,
        sizes=gemm_specs["shape"],
        transpose_a=gemm_specs["transpose_a"],
        transpose_b=gemm_specs["transpose_b"],
        max_iters=100,
        nruns=200,
        nwarmup=200,
        max_nb_configs=10,
        perf_threshold=0.9,
        nb_select_load_tune=3,
        nb_select_pfnb_tune=2,
        target="B70",
    )
    return executed_configs


def infer_params_and_lower(
    mod: ir.Module,
    torch_all_inputs: list[torch.Tensor],
    kernel_metadata: dict,
    benchmark: bool = False,
    payload_func_name: str = "main",
    stop_at_stage: str | None = None,
    params_cache_json: str | None = "kb_params.json",
    enable_tuning: bool = True,
    verbose: int = 0,
) -> ir.Module:
    """
    Infer parameters for the payload module and lower it to LLVM.

    If `stop_at_stage` is provided, the pipeline will stop at that stage and
    raise PipelineInterrupt. If `enable_tuning` is True, the function will
    attempt to tune the parameters for matmul layers.

    Payload function metadata is stored in the `kernel_metadata` dictionary as
    a side effect.

    Args:
        mod: The payload IR module.
        torch_all_inputs: List of torch model parameters and input tensors.
        kernel_metadata: Dictionary to store kernel metadata.
        benchmark: Whether to benchmark the lowered module.
        payload_func_name: Name of the payload function.
        stop_at_stage: Stage at which to stop the lowering pipeline.
        params_cache_json: Path to the JSON file for caching parameters.
        enable_tuning: Whether to enable runtime tuning.
        verbose: Verbosity level.
    Returns:
        The lowered IR module.
    """
    if stop_at_stage == "imported":
        print(mod)
        raise PipelineInterrupt()
    func_metadata, schedule_kind, schedule_params = infer_parameters(
        mod, verbose=verbose
    )
    # store for external use
    kernel_metadata.update(func_metadata)

    matmuls = func_metadata["layers"]["matmul"]
    if enable_tuning and len(matmuls) == 1 and schedule_kind == "mlp":
        # runtime tuning for matmul kernels
        if os.path.isfile(params_cache_json):
            print(f"Loading cached parameters from {params_cache_json}")
            with open(params_cache_json, "r") as f:
                params_dict = json.load(f)
            schedule_params = [params_dict]
        else:
            print(f"Tuning parameters and saving to {params_cache_json}")
            # tune matmul parameters
            configs = tune_matmul_layer(
                matmuls[0],
                mod,
                payload_func_name,
                torch_all_inputs,
                func_metadata["total_flops"],
                func_metadata["read_bytes"],
                func_metadata["write_bytes"],
            )
            schedule_params = [configs[0][1]]
            dump_configs_json(
                schedule_params[0], filename_prefix=params_cache_json.rsplit(".", 1)[0]
            )
    else:
        print(f"Using default parameters for {schedule_kind} schedule")

    if verbose > 2:
        print("Payload module before lowering:")
        print(mod)
    if verbose > 1:
        print(f"Applying '{schedule_kind}' schedule with params:")
        for i, param_dict in enumerate(schedule_params):
            print(f" Parameters for layer {i}:")
            for k, v in param_dict.items():
                print(f"  {k}: {v}")

    lowered_mod = lower_to_llvm(
        mod=mod,
        schedule_kind=schedule_kind,
        schedule_params=schedule_params,
        stop_at_stage=stop_at_stage,
        benchmark=benchmark,
        payload_func_name=payload_func_name,
    )

    if stop_at_stage:
        print(lowered_mod)
        raise PipelineInterrupt()
    elif verbose > 2:
        print("Payload module after full lowering:")
        print(lowered_mod)

    return lowered_mod


def lower_to_llvm(
    mod: ir.Module,
    schedule_kind: str,
    schedule_params: list[dict],
    stop_at_stage: str | None,
    benchmark: bool,
    payload_func_name: str,
) -> ir.Module:
    """Lower payload module to LLVM using the specified schedule and parameters."""
    make_function_callable(mod, payload_func_name)
    if schedule_kind == "mlp":
        schedule = mlp_schedule(
            params=schedule_params,
            payload_func_name=payload_func_name,
            stop_at_stage=stop_at_stage,
        )
    elif schedule_kind == "elemwise":
        schedule = elemwise_schedule(
            params=schedule_params,
            payload_func_name=payload_func_name,
            stop_at_stage=stop_at_stage,
        )
    else:
        raise ValueError(f"Unsupported schedule kind: {schedule_kind}")

    # define lowering pipeline
    schedules = []
    if benchmark:
        schedules.append(Runner.get_bench_wrapper_schedule(payload_func_name))
    schedules.append(schedule)
    if not stop_at_stage or stop_at_stage == "final":
        schedules.append(xegpu_to_binary())
    driver = TransformDriver(schedules=schedules)

    # apply the pipeline to the payload module
    return driver.apply(mod)


def lower_and_execute_benchmark(
    filepath: str,
    level: int,
    id: int,
    datatype: str,
    ctx: ir.Context = None,
    nwarmup: int = 500,
    nruns: int = 500,
    verify: bool = True,
    stop_at_stage: str | None = None,
    verbose: int = 0,
    debug: bool = False,
) -> dict:
    """
    High-level function to lower and execute a KernelBench benchmark.

    Args:
        filepath: Path to the KernelBench benchmark file.
        level: KernelBench level (1, 2, or 3).
        id: Benchmark ID.
        datatype: Data type for the model ('f16' or 'bf16').
        ctx: MLIR context to use. If None, a new context is created.
        nwarmup: Number of warmup runs for benchmarking.
        nruns: Number of runs for benchmarking.
        verify: Whether to verify the result against PyTorch reference.
        stop_at_stage: Stage at which to stop the lowering pipeline.
        verbose: Verbosity level.
        debug: Run without timeout and raise exceptions immediately.
    Returns:
        A dictionary containing benchmark performance metrics.
    """
    assert datatype in ["f16", "bf16"], "Unsupported datatype"
    model_dtype = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[datatype]
    execute = stop_at_stage is None

    # import torch model
    torch_model, torch_inputs, _torch_kwargs = lh_ingress.torch.import_model(
        filepath, model_datatype=model_dtype
    )
    # convert inputs to correct datatype
    torch_inputs = [inp.to(model_dtype) for inp in torch_inputs]

    is_mlp = (
        sum(int(type(module).__name__ == "Linear") for module in torch_model.modules())
        > 1
    )

    # initialize all model weights to random values
    torch.manual_seed(42)  # set random seed
    for param in torch_model.parameters():
        torch.nn.init.uniform_(param, a=-0.5, b=0.5)
        if is_mlp and param.ndim == 2:
            # scale columns to unit norm for more stable correctness verification
            param.data /= param.data.norm(dim=0, keepdim=True) + 1e-6

    if execute:
        # execute torch model on the device
        torch_inputs = [inp.to("xpu") for inp in torch_inputs]
        with torch.no_grad():
            torch_model = torch_model.to("xpu")
            result_ref = torch_model(*torch_inputs).to("cpu")
        torch.xpu.synchronize()

    # compile and execute the model with the MLIR backend
    torch_all_inputs = [*torch_model.parameters(), *torch_inputs]
    # get payload metadata as a side effect
    kernel_metadata = {}
    fn_compile = partial(
        infer_params_and_lower,
        torch_all_inputs=torch_all_inputs,
        kernel_metadata=kernel_metadata,
        benchmark=True,
        payload_func_name="main",
        stop_at_stage=stop_at_stage,
        params_cache_json=f"kb_params_level{level}-{id}.json",
        enable_tuning=execute,
        verbose=2,
    )
    backend = gpu_backend(
        fn_compile,
        device=torch.device("xpu"),
        dialect=TargetDialect.LINALG_ON_TENSORS,
        ir_context=ctx,
        shared_libs=["libmlir_levelzero_runtime.so"],
    )
    torch_model.compile(dynamic=False, backend=backend)
    if not execute:
        try:
            # Get the graph and compile the manually to dump the IR. This works
            # even if LLVM does not have XeGPU support or if the target device
            # is not available.
            gm, _ = dynamo.export(torch_model)(*torch_inputs)
            backend(gm, list(torch_inputs))
        except dynamo.exc.BackendCompilerFailed as e:
            if debug:
                raise e
        return {}

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
            category=FutureWarning,
        )
        result = torch_model(*torch_inputs)
    result = result.to("cpu")
    torch.xpu.synchronize()

    verified = 0
    if verify:
        atol = abs(result_ref).max() * 1e-3
        rtol = 1e-3
        if result_ref.dtype == torch.bfloat16:
            if is_mlp:
                atol = 7e-3
            rtol = 2e-2
        success = torch.allclose(result, result_ref, rtol=rtol, atol=atol)
        verified = 1 if success else 0
        print(f"Verification {'PASSED' if success else 'FAILED'}")
        if verbose > 0 and not success:
            diff = result - result_ref
            # compute max absolute and relative differences to guide tolerance
            max_abs_diff = abs(diff).max()
            print(f"Reference solution range: [{result_ref.min()}, {result_ref.max()}]")
            print(f"Max abs difference: {max_abs_diff:.3e}")

            # compute require atol/rtol for this result keeping rtol/atol fixed
            print(f"atol={atol:.3e}, rtol={rtol:.3e}")
            abs_diff = diff.abs()
            abs_ref = result_ref.abs()
            required_atol = (abs_diff - rtol * abs_ref).max().item()
            required_rtol = ((abs_diff - atol) / abs_ref.clamp_min(1e-30)).max().item()
            print("Possible tolerances for this result to pass:")
            print(f"  atol={required_atol:.3e}, rtol={rtol:.3e}")
            print(f"  atol={atol:.3e}, rtol={required_rtol:.3e}")

    # re-generate random, nonzero weights for performance measurement
    for param in torch_model.parameters():
        torch.nn.init.uniform_(param, a=-0.5, b=0.5)
        # replace zeros with random +/-0.5
        rnd_sign = torch.sign(torch.rand_like(param.data) - 0.5)
        param.data[param.data == 0] = 0.5 * rnd_sign[param.data == 0]

    # benchmark
    if verbose > 0:
        print(f"{nruns=}")
        print(f"{nwarmup=}")
    times = backend.jit_function.benchmark(
        *torch_model.parameters(), *torch_inputs, nruns=nruns, nwarmup=nwarmup
    )
    # compute achieved GFLOPS
    total_flops = kernel_metadata["total_flops"]
    read_bytes = kernel_metadata["read_bytes"]
    write_bytes = kernel_metadata["write_bytes"]
    elapsed, gflops, bandwidth = compute_perf_metrics(
        times, total_flops, read_bytes, write_bytes
    )

    # adjust flop count for elementwise kernels
    flop_factor_dict = {
        (1, 19): 1,  # ReLU
        (1, 20): 2,  # LeakyReLU
        (1, 21): 4,  # Sigmoid
        (1, 22): 4,  # Tanh
        (1, 23): 5,  # Softmax
        (1, 24): 5,  # LogSoftmax
        (1, 25): 5,  # Swish
        (1, 26): 8,  # GELU
        (1, 27): 2,  # SELU
        (1, 28): 2,  # HardSigmoid
        (1, 29): 4,  # Softplus
        (1, 30): 2,  # Softsign
        (1, 31): 2,  # ELU
        (1, 32): 1,  # HardTanh
    }
    if (level, id) in flop_factor_dict:
        factor = flop_factor_dict[(level, id)]
        total_flops *= factor
        gflops *= factor
    print(f"{total_flops=}")

    layers = kernel_metadata["layers"]
    if "matmul" in layers and len(layers["matmul"]) > 0:
        matmuls = layers["matmul"]
        shape_str = " ".join(str(shape) for shape in matmuls)
    elif "elemwise" in layers and len(layers["elemwise"]) > 0:
        elemwise = layers["elemwise"]
        shape_str = elemwise[0]["shape"] if len(elemwise) > 0 else ""
    else:
        shape_str = ""
    entry["flops"] = total_flops
    entry["read_bytes"] = read_bytes
    entry["write_bytes"] = write_bytes
    entry["time"] = elapsed
    entry["throughput"] = gflops
    entry["bandwidth"] = bandwidth
    entry["shapes"] = shape_str
    entry["verified"] = verified
    return entry


def run_experiment(use_timeout: bool = True, timeout: int = 1200, **kwargs) -> dict:
    def ctx_wrapper(*args, **kwargs) -> dict:
        with ir.Context() as ctx, ir.Location.unknown():
            lh_dialects.register_and_load()
            results = lower_and_execute_benchmark(*args, **kwargs, ctx=ctx)
        return results

    if use_timeout:
        exec_func = partial(ctx_wrapper, **kwargs)
        results = run_with_timeout(experiment_func=exec_func, timeout=timeout)
    else:
        results = ctx_wrapper(**kwargs)
    return results


def get_benchmarks(
    pattern: str = "level1/*.py", include: list[int] | None = None
) -> list[tuple[int, Path]]:
    """
    Get a list of KernelBench benchmark files matching the given pattern.

    If `include` is provided, only benchmarks with IDs in the list will be
    returned.

    Returns a list of tuples (benchmark_id, benchmark_path).
    """

    def get_bench_id(bench_file: Path) -> int:
        return int(bench_file.stem.split("_", 1)[0])

    # get local lighthouse path
    lh_root_path = Path(lighthouse.__file__).parent.parent
    print(f"lighthouse root path: {lh_root_path}")
    kb_path = lh_root_path / "third_party/KernelBench/KernelBench/"

    bench_list = [Path(p) for p in glob.glob(str(kb_path / pattern))]
    bench_list = [(get_bench_id(f), f) for f in bench_list]
    bench_list = sorted(bench_list, key=lambda x: x[0])

    if include:
        assert all(i >= 1 and i <= 100 for i in include), "Invalid benchmark ID"
        bench_list = [b for b in bench_list if b[0] in include]

    return bench_list


def parser_cli_args():
    parser = argparse.ArgumentParser(
        description="Execute KernelBench benchmarks with XEGPU lowering."
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="KernelBench level to execute (default: 1)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        nargs="+",
        help="IDs of specific benchmarks to execute (default: all)",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="bf16",
        choices=["f16", "bf16"],
        help="Data type for the model (default: bf16)",
    )
    parser.add_argument(
        "--dump-kernel",
        type=str,
        help="Stop the pipeline at the specified stage",
        choices=[
            "imported",
            "initial",
            "tiled",
            "vectorized",
            "bufferized",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=500,
        help="Number of runs for benchmarking (default: 1000)",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=500,
        help="Number of warmup runs (default: 500)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in single process and raise exceptions immediately.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase output verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parser_cli_args()
    kb_level = args.level
    benchmarks = args.benchmark
    stop_at_stage = args.dump_kernel

    kb_pattern = f"level{kb_level}/*.py"
    bench_list = get_benchmarks(kb_pattern, include=benchmarks)

    if not stop_at_stage:
        csv_file = "out_kernelbench.csv"
        csv_logger = CSVLogger(csv_file, echo_stdout=False, verbose=True)
    else:
        csv_logger = None

    for id, bench_path in bench_list:
        short_path = bench_path.parent.name + "/" + bench_path.name
        print("-" * 80)
        print(f"Executing benchmark: {short_path}", flush=True)
        entry = {
            "id": id,
            "file": bench_path.name,
            "shapes": "",
            "flops": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "executed": 0,
            "verified": 0,
            "time": 0,
            "throughput": 0,
            "bandwidth": 0,
            "error": "",
        }
        try:
            results = run_experiment(
                filepath=str(bench_path),
                level=kb_level,
                id=id,
                datatype=args.datatype,
                nruns=args.nruns,
                nwarmup=args.nwarmup,
                verbose=args.verbose,
                stop_at_stage=stop_at_stage,
                use_timeout=not args.debug,
                debug=args.debug,
            )
            if stop_at_stage:
                continue
            time = results.get("time", 0.0)
            gflops = results.get("throughput", 0.0)
            bw = results.get("bandwidth", 0.0)
            err_msg = results.get("error", "")
            print(
                f"Average execution time: {time:.2f} us {gflops:.2f} GFLOPS {bw:.2f} GB/s",
                flush=True,
            )
            entry.update(results)
            entry["executed"] = 1 if err_msg == "" else 0
            entry["error"] = err_msg
        except Exception as e:
            if args.debug:
                raise e
            print(f"Benchmark {short_path} failed with error: {e}", flush=True)
            entry["error"] = str(e)

        # Store intermediate results
        if csv_logger is not None:
            csv_logger.log(entry)

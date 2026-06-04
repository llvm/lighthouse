# RUN: %PYTHON %s --dry-run --max-iters 1000 | FileCheck %s
# CHECK: Total complexity: 23914845 configurations
# CHECK: Number of executed configurations: 1000

from time import perf_counter
from datetime import timedelta
from itertools import product
import numpy as np
import os
import sys
from csv_logger import CSVLogger

from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.execution.runner import Runner
from lighthouse.schedule.xegpu.mlp_schedule import DPAS
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import check_constraints
from lighthouse.schedule.xegpu import XeGPUSpecs

from matmul import XeGPUMatMul, check_results, cli_parser
from genetic_algorithm import (
    Variable,
    VariableSet,
)
from tune_utils import dump_configs_json, execute_and_log


def run_experiment(
    ab_type: str = "f16",
    c_type: str = "f32",
    nruns: int = None,
    nwarmup: int = None,
    check_result: bool = False,
    has_bias: bool = False,
    has_relu: bool = False,
    accumulate_c: bool = True,
    truncate_c: bool = False,
    **params,
) -> tuple[float, float]:
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMatMul(
            M=params["m"],
            N=params["n"],
            K=params["k"],
            ab_type=ab_type,
            c_type=ab_type if truncate_c else c_type,
            transpose_a=params["transpose_a"],
            transpose_b=params["transpose_b"],
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
            truncate_c=truncate_c,
        )
        pipeline = TransformDriver(wload.schedule_modules(parameters=params))
        payload = pipeline.apply(wload.payload_module())

        runner = Runner(
            payload,
            mem_manager_cls=wload.memory_manager_class,
            shared_libs=wload.shared_libs(),
        )
        if check_result:
            # Setup callback function to copy result from device to host.
            D_host_copy = np.zeros(wload.c_shape, dtype=wload.c_dtype)
            argument_access_callback = Runner.get_gpu_argument_access_callback(
                D_host_copy, arg_index=0
            )
            host_inputs = wload.get_input_arrays(init_int=True)
            runner.execute(
                host_input_buffers=host_inputs,
                payload_function_name=wload.payload_function_name,
                argument_access_callback=argument_access_callback,
            )
            success = check_results(
                wload,
                host_inputs,
                D_host_copy,
                verbose=1,
            )
            if not success:
                raise ValueError("Result mismatch!")
        host_inputs = wload.get_input_arrays()
        if nruns is None and nwarmup is None:
            # first run to estimate cost
            times = runner.benchmark(
                host_input_buffers=host_inputs, nruns=10, nwarmup=10
            )
            # estimate number of runs
            cost = times.mean()
            warmup_target = 0.25
            nwarmup = max(int(warmup_target / cost), 10)
            nruns = 3 * nwarmup
            print(f"{nwarmup=} {nruns=}")
        # benchmark
        times = runner.benchmark(
            host_input_buffers=host_inputs, nruns=nruns, nwarmup=nwarmup
        )

    times *= 1e6  # convert to microseconds
    elapsed = np.mean(times)
    flop_count = wload.get_complexity()[0]
    gflops = flop_count / (elapsed * 1e-6) / 1e9

    return elapsed, gflops


def get_divisors(n: int, min_tile: int = 32, max_tile: int = 256) -> list[int]:
    p = np.ceil(n / max_tile)
    q = n // min_tile
    candidates = n / np.arange(max(p, 1), q + 1)
    candidates = [int(v) for v in candidates if int(v) == v]
    return candidates[::-1]


def divisible_by(a_list: list, b: int) -> list:
    return [a for a in a_list if a % b == 0]


def construct_search_space(
    M: int, N: int, K: int, transpose_a: bool, transpose_b: bool, gpu_specs: XeGPUSpecs
) -> tuple[VariableSet, callable]:
    wg_tile_lim_m = min(max(M // 4, 16), 64), min(M, 256)
    wg_tile_lim_n = min(max(N // 4, 16), 64), min(N, 256)
    sg_tile_lim_m = min(max(M // 8, 16), 32), min(M, 128)
    sg_tile_lim_n = min(max(N // 8, 16), 32), min(N, 128)

    wg_tiles_m = divisible_by(get_divisors(M, *wg_tile_lim_m), DPAS.M)
    wg_tiles_n = divisible_by(get_divisors(N, *wg_tile_lim_n), DPAS.N)
    sg_tiles_m = divisible_by(get_divisors(M, *sg_tile_lim_m), DPAS.M)
    sg_tiles_n = divisible_by(get_divisors(N, *sg_tile_lim_n), DPAS.N)
    k_tiles = divisible_by(get_divisors(K, 16, min(K, 256)), DPAS.K)
    load_tiles = [8, 16, 32]
    prefetch_nb = [1, 2, 3]

    def sample_is_valid(sample_params, verbose=False):
        params = {"m": M, "n": N, "k": K}
        params.update(sample_params)
        return check_constraints(params, gpu_specs, verbose=verbose)

    var_set = VariableSet(
        [
            Variable("wg_m", wg_tiles_m),
            Variable("wg_n", wg_tiles_n),
            Variable("sg_m", sg_tiles_m),
            Variable("sg_n", sg_tiles_n),
            Variable("k_tile", k_tiles),
            Variable("load_a_m", load_tiles),
            Variable("load_a_k", load_tiles),
            Variable("load_b_k", load_tiles),
            Variable("load_b_n", load_tiles),
            Variable("prefetch_a_m", load_tiles),
            Variable("prefetch_a_k", load_tiles),
            Variable("prefetch_b_k", load_tiles),
            Variable("prefetch_b_n", load_tiles),
            Variable("prefetch_a_nb", prefetch_nb),
            Variable("prefetch_b_nb", prefetch_nb),
        ],
        is_valid_fn=sample_is_valid,
    )

    def sample_to_dict(sample: list) -> dict:
        res = {
            "m": M,
            "n": N,
            "k": K,
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
        }
        res.update(var_set.sample_to_dict(sample))
        return res

    return var_set, sample_to_dict


if __name__ == "__main__":
    parser = cli_parser(
        description="Optimize matmul kernel parameters using a exhaustive search."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check validity of combinations but do not execute kernels.",
    )
    parser.add_argument(
        "--target",
        choices=["B70", "B50"],
        default="B70",
        help="Target GPU device.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        help="Maximum number of executed configurations.",
    )
    parser.add_argument(
        "--no-check-result",
        action="store_true",
        help="Skip correctness check.",
    )
    parser.add_argument(
        "--dump-json",
        dest="n_dump_json",
        type=int,
        default=0,
        help="Dump the best n configurations as JSON files.",
    )
    args = parser.parse_args()

    sizes = args.sizes
    transpose_a = args.transpose_a
    transpose_b = args.transpose_b
    has_bias = args.bias
    has_relu = args.relu
    accumulate_c = not args.no_accumulate_c
    truncate_c = not args.truncate_c
    ab_type = "f16"
    c_type = "f32"

    # timeout for kernel execution in seconds
    timeout = 50

    # number of iterations in kernel timing is chosen adaptively
    nwarmup = None
    nruns = None

    # disable IGC compiler cache
    os.environ["NEO_CACHE_PERSISTENT"] = "0"

    if not args.dry_run:
        csv_file = "out_gridsearch.csv"
        csv_logger = CSVLogger(csv_file)

    gpu_specs = XeGPUSpecs.get(args.target)

    var_set, sample_to_dict = construct_search_space(
        *sizes,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        gpu_specs=gpu_specs,
    )
    print(f"Matmul problem size: {sizes}")
    print(f"device={gpu_specs.name}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{transpose_a=}")
    print(f"{transpose_b=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    print(f"{truncate_c=}")
    var_set.print()
    sys.stdout.flush()

    i = 0
    executed_configs = []
    tic = perf_counter()
    for sample in product(*var_set.iterables()):
        params = sample_to_dict(sample)
        if not check_constraints(params, gpu_specs, verbose=False):
            continue

        i += 1
        if args.max_iters is not None and i >= args.max_iters:
            print(f"Reached maximum number of iterations: {args.max_iters}")
            break
        if args.dry_run:
            continue
        time, gflops = execute_and_log(
            run_experiment,
            csv_logger,
            nruns,
            nwarmup,
            params,
            check_result=not args.no_check_result,
            timeout=timeout,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
            truncate_c=truncate_c,
        )
        executed_configs.append((gflops, params))

    duration = perf_counter() - tic
    print(f"Number of executed configurations: {i}")
    print(f"Total duration: {timedelta(seconds=duration)}")

    if args.n_dump_json > 0 and not args.dry_run:
        executed_configs.sort(key=lambda x: x[0], reverse=True)
        best_configs = [c for c in executed_configs[: args.n_dump_json]]
        print("Best configurations found:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")
        sizes_str = "-".join(str(s) for s in sizes)
        relu_str = "_relu" if has_relu else ""
        bias_str = "_bias" if has_bias else ""
        tra_str = "_tra" if transpose_a else ""
        trb_str = "_trb" if transpose_b else ""
        acc_str = "_acc" if accumulate_c else ""
        trunc_str = "_trunc" if truncate_c else ""
        prefix = f"matmul_params_{sizes_str}_{ab_type}-{c_type}{tra_str}{trb_str}{bias_str}{relu_str}{acc_str}{trunc_str}"
        dump_configs_json([p for _, p in best_configs], filename_prefix=prefix)

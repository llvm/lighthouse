# RUN: %PYTHON %s --dry-run | FileCheck %s
# CHECK: Total complexity: 2657205 configurations
# CHECK: Number of executed configurations: 5292

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

from matmul import XeGPUMatMul, execute_and_check, cli_parser
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
    **params,
) -> tuple[float, float]:
    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()

        wload = XeGPUMatMul(
            M=params["m"],
            N=params["n"],
            K=params["k"],
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
        )
        pipeline = TransformDriver(wload.schedule_modules(parameters=params))
        payload = pipeline.apply(wload.payload_module())

        if check_result:
            success = execute_and_check(wload, payload, verbose=1)
            if not success:
                raise ValueError("Result mismatch!")
        runner = Runner(shared_libs=wload.shared_libs())
        if nruns is None and nwarmup is None:
            # first run to estimate cost
            times = runner.benchmark(
                payload,
                host_input_buffers=wload._initial_host_arrays,
                mem_manager_cls=wload.memory_manager_class,
                nruns=10,
                nwarmup=10,
            )
            # estimate number of runs
            cost = times.mean()
            warmup_target = 0.25
            nwarmup = max(int(warmup_target / cost), 10)
            nruns = 3 * nwarmup
            print(f"{nwarmup=} {nruns=}")
        # benchmark
        times = runner.benchmark(
            payload,
            host_input_buffers=wload._initial_host_arrays,
            mem_manager_cls=wload.memory_manager_class,
            nruns=nruns,
            nwarmup=nwarmup,
        )

    times *= 1e6  # convert to microseconds
    elapsed = np.mean(times)
    flop_count = wload.get_complexity()[0]
    gflops = flop_count / (elapsed * 1e-6) / 1e9

    return elapsed, gflops


def check_constraints(params: dict, verbose: bool = False) -> bool:
    def print_reason(msg):
        if verbose:
            print(f"  Invalid: {msg}")

    # hardware constraints
    max_nb_sg_threads = 64
    load_max_rows = 32
    load_max_cols = 16
    pfetch_min_rows = 8
    pfetch_max_rows = 32
    pfetch_min_cols = 16
    pfetch_max_cols = 32

    # heuristics: skip likely suboptimal configurations
    min_nb_threads = 16

    M = params["m"]
    N = params["n"]
    wg_tile_m = params["wg_m"]
    wg_tile_n = params["wg_n"]
    sg_tile_m = params["sg_m"]
    sg_tile_n = params["sg_n"]
    load_tile_a_m = params["load_a_m"]
    load_tile_a_k = params["load_a_k"]
    load_tile_b_k = params["load_b_k"]
    load_tile_b_n = params["load_b_n"]
    prefetch_tile_a_m = params["prefetch_a_m"]
    prefetch_tile_a_k = params["prefetch_a_k"]
    prefetch_tile_b_k = params["prefetch_b_k"]
    prefetch_tile_b_n = params["prefetch_b_n"]
    k_tile = params["k_tile"]

    if M % wg_tile_m != 0:
        print_reason("wg_tile_m does not divide M")
        return False
    if N % wg_tile_n != 0:
        print_reason("wg_tile_n does not divide N")
        return False
    if wg_tile_m % sg_tile_m != 0:
        print_reason("sg_tile_m does not divide wg_tile_m")
        return False
    if wg_tile_n % sg_tile_n != 0:
        print_reason("sg_tile_n does not divide wg_tile_n")
        return False
    if sg_tile_m % DPAS.M != 0:
        print_reason("sg_tile_m not multiple of dpas_m")
        return False
    if sg_tile_n % DPAS.N != 0:
        print_reason("sg_tile_n not multiple of dpas_n")
        return False
    if k_tile % DPAS.K != 0:
        print_reason("k_tile not multiple of dpas_k")
        return False

    # SG level thread layout: [nb_sg_threads_m, nb_sg_threads_n]
    nb_sg_threads_m = wg_tile_m // sg_tile_m
    nb_sg_threads_n = wg_tile_n // sg_tile_n
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > max_nb_sg_threads:
        print_reason("too many sg threads")
        return False
    if nb_sg_threads < min_nb_threads:
        print_reason("too few sg threads")
        return False

    if sg_tile_m % load_tile_a_m != 0:
        print_reason("load_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % load_tile_a_k != 0:
        print_reason("load_tile_a_k does not divide k_tile")
        return False
    if k_tile % load_tile_b_k != 0:
        print_reason("load_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % load_tile_b_n != 0:
        print_reason("load_tile_b_n does not divide sg_tile_n")
        return False
    if load_tile_a_m > load_max_rows:
        print_reason("too large load_tile_a_m")
        return False
    if load_tile_a_k > load_max_cols:
        print_reason("too large load_tile_a_k")
        return False
    if load_tile_b_k > load_max_rows:
        print_reason("too large load_tile_b_k")
        return False
    if load_tile_b_n > load_max_cols:
        print_reason("too large load_tile_b_n")
        return False
    if sg_tile_m % prefetch_tile_a_m != 0:
        print_reason("prefetch_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % prefetch_tile_a_k != 0:
        print_reason("prefetch_tile_a_k does not divide k_tile")
        return False
    if k_tile % prefetch_tile_b_k != 0:
        print_reason("prefetch_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % prefetch_tile_b_n != 0:
        print_reason("prefetch_tile_b_n does not divide sg_tile_n")
        return False
    if prefetch_tile_a_m > pfetch_max_rows:
        print_reason("too large prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k > pfetch_max_cols:
        print_reason("too large prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k > pfetch_max_rows:
        print_reason("too large prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n > pfetch_max_cols:
        print_reason("too large prefetch_tile_b_n")
        return False
    if prefetch_tile_a_m < pfetch_min_rows:
        print_reason("too small prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k < pfetch_min_cols:
        print_reason("too small prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k < pfetch_min_rows:
        print_reason("too small prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n < pfetch_min_cols:
        print_reason("too small prefetch_tile_b_n")
        return False
    if load_tile_a_m % DPAS.M != 0:
        print_reason("load_tile_a_m not multiple of dpas_m")
        return False
    if load_tile_a_k % DPAS.K != 0:
        print_reason("load_tile_a_k not multiple of dpas_k")
        return False
    if load_tile_b_k % DPAS.K != 0:
        print_reason("load_tile_b_k not multiple of dpas_k")
        return False
    if load_tile_b_n % DPAS.N != 0:
        print_reason("load_tile_b_n not multiple of dpas_n")
        return False

    nb_load_b_n = load_tile_b_n // DPAS.N
    if nb_load_b_n > 1:
        # unsupported VNNI layout, loaded tile can only be row-sliced for vnni
        # NOTE this can plausibly be relaxed
        print_reason("invalid load_tile_b_n for VNNI")
        return False

    # prefetch A layout
    nb_prefetch_a_m = wg_tile_m // prefetch_tile_a_m
    nb_prefetch_a_k = k_tile // prefetch_tile_a_k
    if nb_prefetch_a_m * nb_prefetch_a_k > max_nb_sg_threads:
        print_reason("too many prefetch A tiles")
        return False
    if nb_prefetch_a_m * nb_prefetch_a_k < min_nb_threads:
        print_reason("too few prefetch A threads")
        return False

    # prefetch B layout
    nb_prefetch_b_k = k_tile // prefetch_tile_b_k
    nb_prefetch_b_n = wg_tile_n // prefetch_tile_b_n
    if nb_prefetch_b_k * nb_prefetch_b_n > max_nb_sg_threads:
        print_reason("too many prefetch B tiles")
        return False
    if nb_prefetch_b_k * nb_prefetch_b_n < min_nb_threads:
        print_reason("too few prefetch B threads")
        return False

    return True


def get_divisors(n: int, min_tile: int = 32, max_tile: int = 256) -> list[int]:
    p = np.ceil(n / max_tile)
    q = n // min_tile
    candidates = n / np.arange(max(p, 1), q + 1)
    candidates = [int(v) for v in candidates if int(v) == v]
    return candidates[::-1]


def divisible_by(a_list: list, b: int) -> list:
    return [a for a in a_list if a % b == 0]


def construct_search_space(M: int, N: int, K: int):
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
    prefetches = [1]

    def sample_is_valid(sample_params, verbose=False):
        params = {"m": M, "n": N, "k": K}
        params.update(sample_params)
        return check_constraints(params, verbose=verbose)

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
            Variable("prefetch_nb", prefetches),
        ],
        is_valid_fn=sample_is_valid,
    )

    def sample_to_dict(sample: list) -> dict:
        res = {"m": M, "n": N, "k": K}
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
    has_bias = args.bias
    has_relu = args.relu
    accumulate_c = not args.no_accumulate_c
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

    var_set, sample_to_dict = construct_search_space(*sizes)
    print(f"Matmul problem size: {sizes}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    var_set.print()
    sys.stdout.flush()

    i = 0
    executed_configs = []
    tic = perf_counter()
    for sample in product(*var_set.iterables()):
        params = sample_to_dict(sample)
        if not check_constraints(params, verbose=False):
            continue

        i += 1
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
        )
        executed_configs.append((gflops, params))

    duration = perf_counter() - tic
    print(f"Number of executed configurations: {i}")
    print(f"Total duration: {timedelta(seconds=duration)}")

    if args.n_dump_json > 0:
        executed_configs.sort(key=lambda x: x[0], reverse=True)
        best_configs = [c for c in executed_configs[: args.n_dump_json]]
        print("Best configurations found:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values)}")
        sizes_str = "-".join(str(s) for s in sizes)
        relu_str = "_relu" if has_relu else ""
        bias_str = "_bias" if has_bias else ""
        acc_str = "_acc" if accumulate_c else ""
        prefix = (
            f"matmul_params_{sizes_str}_{ab_type}-{c_type}{bias_str}{relu_str}{acc_str}"
        )
        dump_configs_json([p for _, p in best_configs], filename_prefix=prefix)

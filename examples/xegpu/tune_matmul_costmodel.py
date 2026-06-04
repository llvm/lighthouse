# RUN: %PYTHON %s --dry-run --max-iters 10 | FileCheck %s
# CHECK: Total complexity:
# CHECK: Reached maximum number of iterations: 10

from time import perf_counter
from datetime import timedelta
import os
import sys
from csv_logger import CSVLogger

from matmul import cli_parser
from tune_utils import dump_configs_json, execute_and_log
from tune_matmul_gridsearch import run_experiment
from lighthouse.schedule.xegpu.matmul_costmodel import (
    generate_configs,
    expand_configs_with_load_tiles,
    expand_configs_with_prefetch_depth,
)
from lighthouse.schedule.xegpu import XeGPUSpecs

if __name__ == "__main__":
    parser = cli_parser(
        description="""Optimize matmul kernel parameters using a cost model.

Given the matmul configuration (M, N, K, transposes, etc.), the tuning is split
to three phases:

1. Generate valid workgroup, subgroup, k tile size configurations, and sort
   them by the cost model performance estimation. Select best candidates based
   on the `max-nb-configs` parameter and/or the `perf-threshold` parameter for
   actual evaluation. The prefetch tile selection strategy defines how many
   prefetch tile configurations are generated for each (WG, SG, K) candidate
   (options are "first" or "all"). In this phase, A and B load tile sizes are
   set to the DPAS instruction tile sizes. The A and B prefetch depth is set to
   1.

2. `nb-select-load-tune` best configurations from phase 1 are selected for an
   exhaustive search of A and B load tile sizes.

3. `nb-select-pfnb-tune` best configurations from phase 2 are selected for an
   exhaustive search of A and B prefetch depth.
"""
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
        "--nruns",
        type=int,
        default=500,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nruns-auto",
        dest="nruns",
        action="store_const",
        const=None,
        help="Number of warmup and benchmark runs is defined at run time based on kernel execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=500,
        help="Number of warm-up iterations before benchmarking.",
    )
    parser.add_argument(
        "--max-nb-configs",
        type=int,
        default=None,
        help="Maximum number of generated configurations before evaluation.",
    )
    parser.add_argument(
        "--perf-threshold",
        type=float,
        default=0.8,
        help="Relative threshold in [0, 1] used to select candidate configs for load-tile tuning; e.g., 0.8 keeps configs with performance >= 0.8 of the best.",
    )
    parser.add_argument(
        "--prefetch-strategy",
        type=str,
        default="all",
        choices=["all", "first"],
        help="Strategy how to generate prefetch configurations for the first optimization phase.",
    )
    parser.add_argument(
        "--nb-select-load-tune",
        type=int,
        default=4,
        help="Number of top configurations to select for load tile tuning phase.",
    )
    parser.add_argument(
        "--nb-select-pfnb-tune",
        type=int,
        default=8,
        help="Number of top configurations to select for prefetch depth tuning phase.",
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
    truncate_c = args.truncate_c
    ab_type = "f16"
    c_type = "f32"

    # timeout for kernel execution in seconds
    timeout = 50

    nwarmup = args.nwarmup
    nruns = args.nruns
    if nruns is None:
        # Set both to None to enable timing-based heuristics
        nwarmup = None

    # disable IGC compiler cache
    os.environ["NEO_CACHE_PERSISTENT"] = "0"

    if not args.dry_run:
        csv_file = "out_costmodel.csv"
        csv_logger = CSVLogger(csv_file)

    gpu_specs = XeGPUSpecs.get(args.target)

    print(f"\nMatmul problem size: {sizes}")
    print(f"device={gpu_specs.name}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{transpose_a=}")
    print(f"{transpose_b=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    print(f"{truncate_c=}")
    sys.stdout.flush()

    max_nb_configs = args.max_nb_configs
    perf_threshold = args.perf_threshold
    prefetch_strategy = args.prefetch_strategy
    nb_select_load_tune = args.nb_select_load_tune
    nb_select_pfnb_tune = args.nb_select_pfnb_tune

    print("\nTuning parameters:")
    print(f"{max_nb_configs=}")
    print(f"{perf_threshold=}")
    print(f"{prefetch_strategy=}")
    print(f"{nb_select_load_tune=}")
    print(f"{nb_select_pfnb_tune=}")
    print(f"{nwarmup=}")
    print(f"{nruns=}")
    sys.stdout.flush()

    configs = generate_configs(
        *sizes,
        gpu_specs,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        perf_threshold=perf_threshold,
        pf_strategy=prefetch_strategy,
        max_nb_configs=max_nb_configs,
    )
    param_dicts = [params for _, params in configs]

    def eval_configs(param_dicts):
        print(f"Total complexity: {len(param_dicts)} configurations")
        i = 0
        executed_configs = []
        tic = perf_counter()
        for params in param_dicts:
            if args.max_iters is not None and i >= args.max_iters:
                print(f"Reached maximum number of iterations: {args.max_iters}")
                break
            i += 1
            if args.dry_run:
                print(f"Config {i}: {list(params.values())}")
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

        return executed_configs, duration, i

    def run_tuning_phase(param_dicts, executed_configs, phase_msg):
        print(f"\n{phase_msg}")

        executed_configs2, time2, iters2 = eval_configs(param_dicts)
        executed_configs.extend(executed_configs2)
        # sort by performance
        executed_configs.sort(key=lambda x: x[0], reverse=True)

        return time2, iters2

    def summarize_phase(executed_configs, time, iters, message, n_print=10):
        print(f"Time spent in tuning: {timedelta(seconds=time)}")
        print(f"Number of executed configurations: {iters}")
        print(message)
        for gflops, params in executed_configs[:n_print]:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")

    def select_and_expand(executed_configs, nb_select, expand_fn, **expand_kwargs):
        selected_param_dicts = [c[1] for c in executed_configs[:nb_select]]
        return expand_fn(selected_param_dicts, gpu_specs, **expand_kwargs)

    executed_configs = []
    time, iters = run_tuning_phase(
        param_dicts,
        executed_configs,
        "Phase 1: Tuning WG/SG/K tile sizes",
    )
    if args.dry_run:
        exit(0)
    summarize_phase(
        executed_configs,
        time,
        iters,
        "Best configurations found in first tuning phase:",
    )

    if nb_select_load_tune is not None and nb_select_load_tune > 0:
        param_dicts = select_and_expand(
            executed_configs,
            nb_select_load_tune,
            expand_configs_with_load_tiles,
            load_strategy="all",
            exclude_duplicates=True,
        )
        time2, iters2 = run_tuning_phase(
            param_dicts,
            executed_configs,
            f"Phase 2: Tuning load tiles for best {nb_select_load_tune} configurations",
        )
        summarize_phase(
            executed_configs,
            time2,
            iters2,
            "Best configurations found after load tile tuning:",
        )
        time += time2
        iters += iters2

    if nb_select_pfnb_tune is not None and nb_select_pfnb_tune > 0:
        param_dicts = select_and_expand(
            executed_configs,
            nb_select_pfnb_tune,
            expand_configs_with_prefetch_depth,
            max_depth=2,
            exclude_duplicates=True,
        )
        time2, iters2 = run_tuning_phase(
            param_dicts,
            executed_configs,
            f"Phase 3: Tuning prefetch depth for best {nb_select_pfnb_tune} configurations",
        )
        summarize_phase(
            executed_configs,
            time2,
            iters2,
            "Best configurations found after prefetch depth tuning:",
        )
        time += time2
        iters += iters2

    print(f"Total duration: {timedelta(seconds=time)}")
    print(f"Number of executed configurations: {iters}")

    if args.n_dump_json > 0 and not args.dry_run:
        best_configs = [c for c in executed_configs[: args.n_dump_json]]
        sizes_str = "-".join(str(s) for s in sizes)
        sizes_str = "-".join(str(s) for s in sizes)
        relu_str = "_relu" if has_relu else ""
        bias_str = "_bias" if has_bias else ""
        tra_str = "_tra" if transpose_a else ""
        trb_str = "_trb" if transpose_b else ""
        acc_str = "_acc" if accumulate_c else ""
        trunc_str = "_trunc" if truncate_c else ""
        prefix = f"matmul_params_{sizes_str}_{ab_type}-{c_type}{tra_str}{trb_str}{bias_str}{relu_str}{acc_str}{trunc_str}"
        dump_configs_json([p for _, p in best_configs], filename_prefix=prefix)

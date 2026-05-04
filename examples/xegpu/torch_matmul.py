# RUN: %PYTHON %s | FileCheck %s

# TODO: Add a way to optionally enable the test - requires both PyTorch and XPU device.
# REQUIRES: torch-xpu

import argparse
import json
from typing import Optional
from functools import partial
import os
import time

import torch
import torch.nn as nn
from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured

from lighthouse import dialects as lh_dialects
from lighthouse import schedule as lh_schedule
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.utils.mlir import get_mlir_library_path
from lighthouse.schedule.xegpu import mlp_schedule, xegpu_to_binary
from lighthouse.ingress.torch import gpu_backend, TargetDialect

import parameter_selector


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


def shared_libs() -> list[str]:
    lib_dir = get_mlir_library_path()
    libs = ["libmlir_levelzero_runtime.so"]

    found_libs = []
    for so_file in libs:
        so_path = os.path.join(lib_dir, so_file)
        if not os.path.isfile(so_path):
            raise ValueError(f"Could not find shared library {so_path}")
        found_libs.append(so_path)
    return found_libs


def schedule_modules(parameters: Optional[dict] = None) -> list[ir.Module]:
    assert parameters is not None, "Schedule parameters must be provided"
    schedules = []

    with lh_schedule.schedule_boilerplate() as (sched, named_seq):
        func_op = structured.MatchOp.match_op_names(named_seq.bodyTarget, ["func.func"])
        transform.apply_registered_pass(
            transform.any_op_t(), func_op, "llvm-request-c-wrappers"
        )
        transform.yield_()
    schedules.append(sched)

    schedules.append(
        mlp_schedule(
            params=[parameters],
        )
    )

    schedules.append(xegpu_to_binary())

    return schedules


def lower_to_llvm(module: ir.Module, parameters: Optional[dict] = None) -> ir.Module:
    pipeline = TransformDriver(schedule_modules(parameters=parameters))
    payload = pipeline.apply(module)
    return payload


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
        default=0,
        help="Number of runs to average the execution time.",
    )
    parser.add_argument(
        "--nwarmup",
        type=int,
        default=20,
        help="Number of warm-up iterations before benchmarking.",
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

    for param_key, v in params.items():
        if v is None:
            raise ValueError(
                f"Parameter {param_key} is not set. Please provide it via CLI or JSON file."
            )

    with ir.Context() as ctx, ir.Location.unknown():
        lh_dialects.register_and_load()

        xpu_device = torch.device("xpu")
        a = torch.randn([m, k], device=xpu_device, dtype=torch.float16)
        b = torch.randn([k, n], device=xpu_device, dtype=torch.float16)

        model = Model()
        out_ref = model(a, b)
        torch.xpu.synchronize()

        fn_compile = partial(lower_to_llvm, parameters=params)
        model.compile(
            dynamic=False,
            backend=gpu_backend(
                fn_compile,
                device=xpu_device,
                dialect=TargetDialect.LINALG_ON_TENSORS,
                ir_context=ctx,
                shared_libs=shared_libs(),
            ),
        )
        out = model(a, b)
        torch.xpu.synchronize()

        is_match = torch.allclose(out_ref, out, rtol=0.01, atol=0.01)

        # CHECK: Compile function - result match: True
        print(f"Compile function - result match: {is_match}")

        if args.nruns > 0:
            # Warmup
            for _ in range(args.nwarmup):
                model(a, b)

            # Benchmark loop.
            start = time.perf_counter_ns()
            for i in range(args.nruns):
                # MLIR synchronizes internally.
                # No need for extra synchronization here.
                model(a, b)
            end = time.perf_counter_ns()

            elapsed = (end - start) / args.nruns / 1e3  # Convert to μs
            flop_count = 2 * m * n * k
            gflops = flop_count / (elapsed * 1e-6) / 1e9

            def list2str(a):
                return ",".join(map(str, a))

            ab_type = str(a.dtype)
            c_type = str(out.dtype)
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

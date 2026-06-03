# RUN: python %s --ci | FileCheck %s
# RUN: python %s --ci --torch-compile | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

import argparse
import os
import re
import subprocess
import platform
from pathlib import Path

import yaml

script_path = Path(__file__).parent
project_root = script_path.parent.parent
kb_program = project_root / "tools" / "kernel-bench"
kb_default_pipeline = kb_program.parent / "kernel-bench.yaml"
kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"
level1_yaml_path = script_path / "level1.yaml"
level2_yaml_path = script_path / "level2.yaml"


class TargetInfo:
    """
    Struct to hold target architecture and feature information.
    Since this is used in a JIT context, we can safely assume the host
    architecture is the target architecture if not specified.
    """

    def __init__(self, arch: str = None, feature: str = None):
        if arch is not None:
            self.arch = arch
        else:
            self.arch = platform.machine()
        self.feature = feature


def get_pipeline_file(name: str, dtype: str, target: TargetInfo) -> Path:
    """
    Returns the appropriate pipeline file for a given kernel.
    """
    # If no name is provided, return the default pipeline.
    if not name:
        return kb_default_pipeline

    # First check if there's a pipeline file specific to the target features.
    if target.feature:
        pipeline = script_path / f"schedules/{target.feature}/{name}/{dtype}.yaml"
        if pipeline.exists():
            return pipeline

    # Second, check if there's a pipeline file specific to the target architecture.
    if target.arch:
        pipeline = script_path / f"schedules/{target.arch}/{name}/{dtype}.yaml"
        if pipeline.exists():
            return pipeline

    # Otherwise, just return the safe option
    return kb_default_pipeline


def get_tests(args: argparse.Namespace) -> list[dict]:
    """
    Returns the list of tests to be executed.
    """
    tests = []
    with open(level1_yaml_path) as f:
        tests = yaml.safe_load(f)
    with open(level2_yaml_path) as f:
        tests += yaml.safe_load(f)

    test_list = []
    for test in tests:
        # If a specific kernel is specified, only include that kernel
        if args.kernel and not test["kernel"].startswith(args.kernel):
            continue
        # CI mode runs fewer tests for faster feedback
        if args.ci and len(test_list) >= 5:
            break
        # Smoke tests run on the simplest lowering
        if args.smoke_test:
            test["pipeline"] = str(kb_default_pipeline)

        test_list.append(
            {
                "kernel": test["kernel"],
                "input_shapes": ",".join(
                    f"{shape}x{args.dtype}x{init}"
                    for shape, init in zip(
                        test["input_shapes"], test["initializations"]
                    )
                ),
                "output_shape": f"{test['output_shape']}x{args.dtype}x0",
                "init_args": test.get("init_args", "None"),
                "gflops": eval(test["gflops"])
                if "gflops" in test and args.benchmark
                else None,
                "pipeline": str(
                    get_pipeline_file(
                        test.get("pipeline", ""),
                        args.dtype,
                        TargetInfo(args.target, args.feature),
                    )
                ),
                "warning": test.get("warning", None),
            }
        )
    return test_list


def get_flops_per_second(stdout: str, gflops: float) -> float:
    for line in stdout.splitlines():
        match = re.search(r"([0-9.e-]+) seconds", line)
        if match:
            seconds = float(match.group(1))
            return gflops / seconds
    return 0.0


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        description="""Kernel Bench testing & benchmarking."""
    )
    Parser.add_argument(
        "--dtype",
        type=str,
        default="f32",
        help="Data type. Default f32.",
    )
    Parser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        help="Whether to run the benchmark.",
    )
    Parser.add_argument(
        "--ci",
        action=argparse.BooleanOptionalAction,
        help="Enable CI mode (faster run, fewer kernels). Incompatible with --smoke-test.",
    )
    Parser.add_argument(
        "--infer-shapes",
        action=argparse.BooleanOptionalAction,
        help="Enable shape inference mode. Default is to use values in YAML file.",
    )
    Parser.add_argument(
        "--kernel",
        type=str,
        help="Specify a particular kernel to run.",
    )
    Parser.add_argument(
        "--print-mlir-after-all",
        action=argparse.BooleanOptionalAction,
        help="Whether to print the MLIR module after all stages. Default is False.",
    )
    Parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        help="Runs every kernel with loops lowering to pipe clean.",
    )
    Parser.add_argument(
        "--target",
        type=str,
        help="Specify a particular target architecture (x86_64, arm64, etc.). Default is to auto-detect.",
    )
    Parser.add_argument(
        "--feature",
        type=str,
        help="Specify a particular target feature (amx, avx512, etc.).",
    )
    args, unknown_args = Parser.parse_known_args()
    if args.smoke_test and args.ci:
        print("\nERROR: Smoke test and CI mode are incompatible.\n")
        Parser.print_help()
        exit(1)

    tests = get_tests(args)
    if len(tests) == 0:
        if args.kernel:
            print(
                f"No tests found matching '{args.kernel}'. Please check your arguments."
            )
        else:
            print("No tests to run. Please check your arguments.")
        exit(0)

    for test in tests:
        kb_kernel = kb_path / test["kernel"]
        command_line = []

        # Pin the process to avoid migration
        if args.benchmark:
            core = min(os.cpu_count() - 1, 3)
            command_line += ["taskset", "-c", f"{core}"]

        command_line += [
            str(kb_program),
            str(kb_kernel),
            "--pipeline",
            test["pipeline"],
            "--dtype",
            args.dtype,
        ]

        # Benchmark mode.
        if args.benchmark:
            command_line += ["--benchmark"]

        # Shape inference or from args.
        if not args.infer_shapes:
            command_line += [
                "--input-shapes",
                test["input_shapes"],
                "--output-shape",
                test["output_shape"],
                "--init-args",
                test["init_args"],
            ]

        # For debugging, prefer not to capture output.
        if args.print_mlir_after_all:
            command_line += ["--print-mlir-after-all"]

        # If any extra args are provided, add them to the command line.
        if unknown_args:
            command_line += unknown_args

        # Print out before we run the test.
        if test.get("warning"):
            print(f"WARNING: {test['warning']}")
        print(f"Running command: {' '.join(command_line)}", flush=True)

        # While debugging kernels, it's useful to see the output as it comes.
        # Note: GFLOPS can't be shown if the output is not captured.
        capture_output = True
        if args.print_mlir_after_all and not args.ci:
            capture_output = False

        result = subprocess.run(
            command_line,
            capture_output=capture_output,
            text=True,
        )

        # If output is captured, print it out, including benchmark results if applicable.
        if capture_output:
            print("STDOUT:")
            print(result.stdout)
            # Only show "performance" if gflops count is available.
            if args.benchmark and test["gflops"] is not None:
                flops_per_second = get_flops_per_second(result.stdout, test["gflops"])
                if flops_per_second > 0:
                    print(f"Performance: {flops_per_second:.2f} GFLOPS")
            # Otherwise just keep the timer and let the user calculate GFLOPS themselves.
            elif args.benchmark:
                print("Performance: GFLOPS data not available for this test.")

            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}", flush=True)

        # Only stop on failure on normal runs.
        # Smoke tests try to run as much as possible.
        if not args.smoke_test:
            assert result.returncode == 0, "Execution failed"

# CHECK: 1_Square_matrix_multiplication_.py
# CHECK: Success: The output of the compiled model matches the reference output.

# CHECK: 2_Standard_matrix_multiplication_.py
# CHECK: Success: The output of the compiled model matches the reference output.

# CHECK: 3_Batched_matrix_multiplication.py
# CHECK: Success: The output of the compiled model matches the reference output.

# CHECK: 4_Matrix_vector_multiplication_.py
# CHECK: Success: The output of the compiled model matches the reference output.

# CHECK: 5_Matrix_scalar_multiplication.py
# CHECK: Success: The output of the compiled model matches the reference output.

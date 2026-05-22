# RUN: python %s --ci | FileCheck %s
# RUN: python %s --ci --torch-compile | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

import argparse
import re
import subprocess
import platform
from pathlib import Path

import yaml

script_path = Path(__file__).parent
project_root = script_path.parent.parent.parent
kb_program = project_root / "tools" / "kernel_bench"
kb_default_pipeline = kb_program.parent / "kernel_bench.yaml"
kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"
level1_yaml_path = script_path / "level1.yaml"
level2_yaml_path = script_path / "level2.yaml"


def get_pipeline_file(name: str, dtype: str) -> Path:
    """
    Returns the appropriate pipeline file for a given kernel.
    """
    arch = platform.machine()

    # If the pipeline file exists for the given name and dtype
    if name:
        pipeline = script_path / f"schedules/{arch}/{name}/{dtype}.yaml"
        if pipeline.exists():
            return pipeline

    # Otherwise, just return the safe option
    return kb_default_pipeline


def get_tests(args: argparse.Namespace) -> list[dict]:
    """
    Returns the list of tests to be executed.
    """
    if args.ci:
        print(
            "Running in CI mode: fewer tests, no bf16, no benchmarking for faster feedback"
        )
        args.bf16 = False  # Disable bf16 tests in CI for faster feedback
        args.benchmark = False  # Disable benchmarking in CI for faster feedback

    tests = []
    with open(level1_yaml_path) as f:
        tests = yaml.safe_load(f)
    with open(level2_yaml_path) as f:
        tests += yaml.safe_load(f)

    test_list = []
    for test in tests:
        # If a specific test is specified, only include that test
        if args.test and not test["kernel"].startswith(args.test):
            continue
        # CI mode runs fewer tests for faster feedback
        if args.ci and len(test_list) >= 5:
            break
        # Smoke tests run on the simplest lowering
        if args.smoke_test:
            test["pipeline"] = str(kb_default_pipeline)
        for dtype in test["dtypes"]:
            if not args.bf16 and dtype == "bf16":
                continue
            test_list.append(
                {
                    "kernel": test["kernel"],
                    "input_shapes": ",".join(
                        f"{shape}x{dtype}x{init}"
                        for shape, init in zip(
                            test["input_shapes"], test["initializations"]
                        )
                    ),
                    "output_shape": f"{test['output_shape']}x{dtype}x0",
                    "gflops": eval(test["gflops"])
                    if "gflops" in test and args.benchmark
                    else None,
                    "pipeline": str(get_pipeline_file(test.get("pipeline", ""), dtype)),
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
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        help="Whether to run the benchmark.",
    )
    Parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        help="Enable bf16 precision kernels.",
    )
    Parser.add_argument(
        "--ci",
        action=argparse.BooleanOptionalAction,
        help="Enable CI mode (faster run, fewer kernels). Incompatible with --smoke-test.",
    )
    Parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        help="Enable TorchScript compilation. Default is False.",
    )
    Parser.add_argument(
        "--test",
        type=str,
        help="Specify a particular test to run.",
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
    args = Parser.parse_args()
    if args.smoke_test and args.ci:
        print("\nERROR: Smoke test and CI mode are incompatible.\n")
        Parser.print_help()
        exit(1)

    tests = get_tests(args)
    if len(tests) == 0:
        if args.test:
            print(
                f"No tests found matching '{args.test}'. Please check your arguments."
            )
        else:
            print("No tests to run. Please check your arguments.")
        exit(0)

    for test in tests:
        kb_kernel = kb_path / test["kernel"]
        command_line = [
            str(kb_program),
            str(kb_kernel),
            "--pipeline",
            test["pipeline"],
            "--seed=42",
        ]
        # Benchmarks only if there's data to calculate FLOPS.
        benchmark = args.benchmark and test.get("gflops") is not None
        if benchmark:
            command_line += ["--benchmark"]

        # We allow torch.compile to pick its own shapes (unless it's CI).
        if args.torch_compile:
            command_line += ["--torch-compile"]

        # TODO: Implement auto-shapes for non-compile mode as well.
        if args.ci or not args.torch_compile:
            command_line += [
                "--input-shapes",
                test["input_shapes"],
                "--output-shape",
                test["output_shape"],
            ]

        # Smoke tests / CI don't print outputs.
        if not args.smoke_test and not args.ci:
            command_line += ["--print-output"]

        # For debugging, prefer not to capture output.
        if args.print_mlir_after_all:
            command_line += ["--print-mlir-after-all"]

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
            if benchmark:
                flops_per_second = get_flops_per_second(result.stdout, test["gflops"])
                if flops_per_second > 0:
                    print(f"Performance: {flops_per_second:.2f} GFLOPS")

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

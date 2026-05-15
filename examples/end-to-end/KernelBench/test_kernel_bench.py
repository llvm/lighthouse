# RUN: python %s --ci | FileCheck %s

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
yaml_path = script_path / "tests.yaml"


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
    with open(yaml_path) as f:
        tests = yaml.safe_load(f)

    test_list = []
    for test in tests:
        for dtype in test["dtypes"]:
            if not args.bf16 and dtype == "bf16":
                continue
            # If a specific test is specified, only include that test
            if args.test and not test["kernel"].startswith(args.test):
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
                    "gflops": test["gflops"]
                    if "gflops" in test and args.benchmark
                    else None,
                    "pipeline": str(get_pipeline_file(test.get("pipeline", ""), dtype)),
                    "warning": test.get("warning", None),
                }
            )
            # CI mode runs fewer tests for faster feedback
            if args.ci and len(test_list) >= 5:
                return test_list
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
        help="Enable CI mode (faster run, fewer kernels).",
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
    args = Parser.parse_args()
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
            "--input-shapes",
            test["input_shapes"],
            "--output-shape",
            test["output_shape"],
            "--pipeline",
            test["pipeline"],
            "--print-tensor=1",
            "--seed=42",
        ]
        benchmark = args.benchmark and test.get("gflops") is not None
        if benchmark:
            command_line += ["--benchmark"]
        if args.print_mlir_after_all:
            command_line += ["--print-mlir-after-all"]
        if test.get("warning"):
            print(f"WARNING: {test['warning']}")
        print(f"Running command: {' '.join(command_line)}")

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

        print(f"Return code: {result.returncode}")
        assert result.returncode == 0, "Execution failed"

# CHECK: 1_Square_matrix_multiplication_.mlir
# CHECK: 0.3745{{.*}} 0.9507{{.*}} 0.7319{{.*}} ... 0.2973{{.*}} 0.9243{{.*}} 0.9710{{.*}}
# CHECK: 0.7201{{.*}} 0.9926{{.*}} 0.1208{{.*}} ... 0.1742{{.*}} 0.3485{{.*}} 0.6436{{.*}}

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 249.78{{.*}} 260.13{{.*}} 249.36{{.*}} ... 261.10{{.*}} 260.49{{.*}} 257.09{{.*}}
# CHECK: 243.56{{.*}} 250.91{{.*}} 252.38{{.*}} ... 260.40{{.*}} 261.56{{.*}} 256.24{{.*}}

# CHECK: 3_Batched_matrix_multiplication.mlir
# CHECK: 5.2403{{.*}} 7.7905{{.*}} 6.0769{{.*}} ... 7.8579{{.*}} 6.8890{{.*}} 6.6193{{.*}}
# CHECK: 9.0407{{.*}} 6.3299{{.*}} 5.2003{{.*}} ... 6.2594{{.*}} 6.2980{{.*}} 5.9807{{.*}}

# CHECK: 4_Matrix_vector_multiplication_.mlir
# CHECK: 264.86{{.*}}
# CHECK: 265.12{{.*}}

# CHECK: 5_Matrix_scalar_multiplication.mlir
# CHECK: 0.1750{{.*}} 0.4442{{.*}} 0.3420{{.*}} ... 0.1389{{.*}} 0.4319{{.*}} 0.4538{{.*}}
# CHECK: 0.3365{{.*}} 0.4638{{.*}} 0.0564{{.*}} ... 0.0814{{.*}} 0.1628{{.*}} 0.3007{{.*}}

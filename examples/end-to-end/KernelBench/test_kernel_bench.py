# RUN: python %s | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

import argparse
import re
import subprocess
import platform
from pathlib import Path

script_path = Path(__file__).parent
project_root = script_path.parent.parent.parent
kb_program = project_root / "tools" / "kernel_bench"
kb_default_pipeline = kb_program.parent / "kernel_bench.yaml"
kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"


def get_pipeline_file(kernel_name: str, dtype: str) -> Path:
    """
    Returns the appropriate pipeline file for a given kernel.
    """
    arch = platform.machine()
    if arch != "x86_64":
        return kb_default_pipeline

    # Level 1 matmuls should use the same pipelines
    if kernel_name.startswith("level1") and "matrix_multiplication" in kernel_name:
        pipeline = script_path / f"schedules/{arch}/matmul/{dtype}.yaml"
        if pipeline.exists():
            return pipeline

    # Otherwise, just return the safe option
    return kb_default_pipeline


tests = [
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": ["1024x1024", "1024x1024"],
        "initializations": ["rnd", "id"],
        "output_shape": "1024x1024",
        "dtypes": ["f32", "bf16"],
        "gflops": (1024 * 1024 * 1024 * 2) / 1e9,
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": ["512x1024", "1024x512"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "512x512",
        "dtypes": ["f32", "bf16"],
        "gflops": (512 * 1024 * 512 * 2) / 1e9,
    },
]


def get_tests(args: argparse.Namespace) -> list[dict]:
    """
    Returns the list of tests to be executed.
    """
    test_list = []
    for test in tests:
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
                    "gflops": test["gflops"]
                    if "gflops" in test and args.benchmark
                    else None,
                    "pipeline": str(get_pipeline_file(test["kernel"], dtype)),
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
    args = Parser.parse_args()

    for test in get_tests(args):
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
        benchmark = test.get("gflops") is not None
        if benchmark:
            command_line += ["--benchmark"]
        print(f"Running command: {' '.join(command_line)}")
        result = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
        )

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

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 249.78{{.*}} 260.13{{.*}} 249.36{{.*}} ... 261.10{{.*}} 260.49{{.*}} 257.09{{.*}}
# CHECK: 243.56{{.*}} 250.91{{.*}} 252.38{{.*}} ... 260.40{{.*}} 261.56{{.*}} 256.24{{.*}}

# CHECK-NOT: Execution failed

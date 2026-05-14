# RUN: python %s --ci | FileCheck %s

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


tests = [
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": ["1024x1024", "1024x1024"],
        "initializations": ["rnd", "id"],
        "output_shape": "1024x1024",
        "dtypes": ["f32", "bf16"],
        "gflops": (1024 * 1024 * 1024 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": ["512x1024", "1024x512"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "512x512",
        "dtypes": ["f32", "bf16"],
        "gflops": (512 * 1024 * 512 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/3_Batched_matrix_multiplication.py",
        "input_shapes": ["4x64x32", "4x32x64"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "4x64x64",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/4_Matrix_vector_multiplication_.py",
        "input_shapes": ["1024x1024", "1024x1"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "1024x1",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/5_Matrix_scalar_multiplication.py",
        "input_shapes": ["1024x1024", "1"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "1024x1024",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/6_Matmul_with_large_K_dimension_.py",
        "input_shapes": ["256x524288", "524288x256"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "256x256",
        "dtypes": ["f32", "bf16"],
        "gflops": (256 * 524288 * 256 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/7_Matmul_with_small_K_dimension_.py",
        "input_shapes": ["32768x64", "64x32768"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "32768x32768",
        "dtypes": ["f32", "bf16"],
        "gflops": (32768 * 64 * 32768 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/8_Matmul_with_irregular_shapes_.py",
        "input_shapes": ["8205x2949", "2949x5921"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "8205x5921",
        "dtypes": ["f32", "bf16"],
        "gflops": (8205 * 2949 * 5921 * 2) / 1e9,
        "pipeline": "matmul",
    },
    # too many tiles provided, expected at most 3 found 4
    {
        "kernel": "level1/9_Tall_skinny_matrix_multiplication_.py",
        "input_shapes": ["1024x32", "32x1024"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "1024x1024",
        "dtypes": ["f32", "bf16"],
        # "gflops": (1024 * 32 * 1024 * 2) / 1e9,
        # "pipeline": "matmul",
    },
    {
        "kernel": "level1/10_3D_tensor_matrix_multiplication.py",
        "input_shapes": ["16x1024x2048", "2048x768"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "16x1024x768",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/11_4D_tensor_matrix_multiplication.py",
        "input_shapes": ["8x256x512x256", "256x768"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "8x256x512x768",
        "dtypes": ["f32", "bf16"],
    },
    # level1/12_Matmul_with_diagonal_matrices_.py
    # torch_mlir.compiler_utils.TorchMlirCompilerError:
    # Lowering TorchFX IR -> Torch Backend IR failed with the following diagnostics:
    # python exception: Failure while executing pass pipeline
    {
        "kernel": "level1/12_Matmul_with_diagonal_matrices_.py",
        "input_shapes": ["4096", "4096x4096"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "4096x4096",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/13_Matmul_for_symmetric_matrices.py",
        "input_shapes": ["4096x4096", "4096x4096"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "4096x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (4096 * 4096 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    # level1/14_Matmul_for_upper_triangular_matrices.py
    # LLVM ERROR: operation destroyed but still has uses
    {
        "kernel": "level1/14_Matmul_for_upper_triangular_matrices.py",
        "input_shapes": ["4096x4096", "4096x4096"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "4096x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (4096 * 4096 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    # level1/15_Matmul_for_lower_triangular_matrices.py
    # LLVM ERROR: operation destroyed but still has uses
    {
        "kernel": "level1/15_Matmul_for_lower_triangular_matrices.py",
        "input_shapes": ["4096x4096", "4096x4096"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "4096x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (4096 * 4096 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/16_Matmul_with_transposed_A.py",
        "input_shapes": ["8192x2048", "8192x4096"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "2048x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (2048 * 8192 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/17_Matmul_with_transposed_B.py",
        "input_shapes": ["2048x8192", "4096x8192"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "2048x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (2048 * 8192 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    {
        "kernel": "level1/18_Matmul_with_transposed_both.py",
        "input_shapes": ["8192x2048", "4096x8192"],
        "initializations": ["rnd", "rnd"],
        "output_shape": "2048x4096",
        "dtypes": ["f32", "bf16"],
        "gflops": (2048 * 8192 * 4096 * 2) / 1e9,
        "pipeline": "matmul",
    },
    # All Element-wise kernels below fail with the same error:
    # LLVM ERROR: operation destroyed but still has uses
    {
        "kernel": "level1/19_ReLU.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/20_LeakyReLU.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/21_Sigmoid.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/22_Tanh.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/23_Softmax.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/24_LogSoftmax.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/25_Swish.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/26_GELU_.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/27_SELU_.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/28_HardSigmoid.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/29_Softplus.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/30_Softsign.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/31_ELU.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/32_HardTanh.py",
        "input_shapes": ["4096x393216"],
        "initializations": ["rnd"],
        "output_shape": "4096x393216",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/33_BatchNorm.py",
        "input_shapes": ["64x64x512x512"],
        "initializations": ["rnd"],
        "output_shape": "64x64x512x512",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/34_InstanceNorm.py",
        "input_shapes": ["112x64x512x512"],
        "initializations": ["rnd"],
        "output_shape": "112x64x512x512",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/35_GroupNorm_.py",
        "input_shapes": ["112x64x512x512"],
        "initializations": ["rnd"],
        "output_shape": "112x64x512x512",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/36_RMSNorm_.py",
        "input_shapes": ["112x64x512x512"],
        "initializations": ["rnd"],
        "output_shape": "112x64x512x512",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/37_FrobeniusNorm_.py",
        "input_shapes": ["112x64x512x512"],
        "initializations": ["rnd"],
        "output_shape": "112x64x512x512",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/38_L1Norm_.py",
        "input_shapes": ["32768x65535"],
        "initializations": ["rnd"],
        "output_shape": "32768x65535",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/39_L2Norm_.py",
        "input_shapes": ["32768x65535"],
        "initializations": ["rnd"],
        "output_shape": "32768x65535",
        "dtypes": ["f32", "bf16"],
    },
    {
        "kernel": "level1/40_LayerNorm.py",
        "input_shapes": ["16x64x256x256"],
        "initializations": ["rnd"],
        "output_shape": "16x64x256x256",
        "dtypes": ["f32", "bf16"],
    },
]


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

    test_list = []
    for test in tests:
        for dtype in test["dtypes"]:
            if not args.bf16 and dtype == "bf16":
                continue
            # If a specific test is specified, only include that test
            if args.test and test["kernel"] != args.test:
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

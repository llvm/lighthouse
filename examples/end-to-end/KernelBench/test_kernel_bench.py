# RUN: python %s | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

import re
import subprocess
import platform
from pathlib import Path

script_path = Path(__file__).parent
project_root = script_path.parent.parent.parent
kb_program = project_root / "tools" / "kernel_bench"
kb_default_pipeline = kb_program.parent / "kernel_bench.yaml"
kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"
arch = platform.machine()
tests = [
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": "1024x1024xf32xrnd,1024x1024xf32xid",
        "output_shape": "1024x1024xf32x0",
        "gflops": (1024 * 1024 * 1024 * 2) / 1e9,
        "pipeline": f"{script_path}/cpu_matmul.yaml"
        if arch == "x86_64"
        else str(kb_default_pipeline),
    },
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": "32x32xbf16xrnd,32x32xbf16xid",
        "output_shape": "32x32xbf16x0",
        "pipeline": str(kb_default_pipeline),
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": "512x1024xf32xrnd,1024x512xf32xrnd",
        "output_shape": "512x512xf32x0",
        "gflops": (512 * 1024 * 512 * 2) / 1e9,
        "pipeline": f"{script_path}/cpu_matmul.yaml"
        if arch == "x86_64"
        else str(kb_default_pipeline),
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": "8x16xbf16xrnd,16x8xbf16xrnd",
        "output_shape": "8x8xbf16x0",
        "pipeline": str(kb_default_pipeline),
    },
]


def get_flops_per_second(stdout: str, gflops: float) -> float:
    for line in stdout.splitlines():
        match = re.search(r"([0-9.e-]+) seconds", line)
        if match:
            seconds = float(match.group(1))
            return gflops / seconds
    return 0.0


if __name__ == "__main__":
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
        if "gflops" in test:
            command_line += ["--benchmark"]
        print(f"Running command: {' '.join(command_line)}")
        result = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)
        if "gflops" in test:
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
# CHECK: Performance: {{.*}} GFLOPS

# CHECK-NOT: Execution failed

# CHECK: 1_Square_matrix_multiplication_.mlir
# CHECK: 0.375 0.949219 0.730469 ... 0.0463867 0.609375 0.170898
# CHECK: 0.271484 0.589844 0.361328 ... 0.296875 0.925781 0.972656

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 249.78{{.*}} 260.13{{.*}} 249.36{{.*}} ... 261.10{{.*}} 260.49{{.*}} 257.09{{.*}}
# CHECK: 243.56{{.*}} 250.91{{.*}} 252.38{{.*}} ... 260.40{{.*}} 261.56{{.*}} 256.24{{.*}}
# CHECK: Performance: {{.*}} GFLOPS

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 3.125 3.76562 4.53125 4.40625 4.4375 3.26562 3.53125 3.9375
# CHECK: 5.03125 5.3125 5.8125 4.8125 4.75 4.34375 5.3125 5.5625

# CHECK-NOT: Execution failed

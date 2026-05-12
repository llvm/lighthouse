# RUN: python %s | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

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
        "input_shapes": "32x32xf32xrnd,32x32xf32xid",
        "output_shape": "32x32xf32x0",
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
        "input_shapes": "8x16xf32xrnd,16x8xf32xrnd",
        "output_shape": "8x8xf32x0",
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
        print(f"Running command: {' '.join(command_line)}")
        result = subprocess.run(
            command_line,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        print(f"Return code: {result.returncode}")
        assert result.returncode == 0, "Execution failed"

# CHECK: 1_Square_matrix_multiplication_.mlir
# CHECK  0.3745{{.*}} 0.9507{{.*}} 0.7319{{.*}} ... 0.0464{{.*}} 0.6075{{.*}} 0.1705{{.*}}
# CHECK: 0.2721{{.*}} 0.5902{{.*}} 0.3609{{.*}} ... 0.2973{{.*}} 0.9243{{.*}} 0.9710{{.*}}

# CHECK-NOT: Execution failed

# CHECK: 1_Square_matrix_multiplication_.mlir
# CHECK  0.375 0.949219 0.730469 ... 0.0463867 0.609375 0.170898
# CHECK: 0.271484 0.589844 0.361328 ... 0.296875 0.925781 0.972656

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 3.1209{{.*}} 3.7697{{.*}} 4.5365{{.*}} 4.3976{{.*}} 4.4506{{.*}} 3.2665{{.*}} 3.5362{{.*}}
# CHECK: 5.0367{{.*}} 5.3128{{.*}} 5.8109{{.*}} 4.8100{{.*}} 4.7435{{.*}} 4.3557{{.*}} 5.3115{{.*}}

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 3.125 3.76562 4.53125 4.40625 4.4375 3.26562 3.53125 3.9375
# CHECK: 5.03125 5.3125 5.8125 4.8125 4.75 4.34375 5.3125 5.5625

# CHECK-NOT: Execution failed

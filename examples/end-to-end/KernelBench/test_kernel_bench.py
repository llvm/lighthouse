# RUN: python %s | FileCheck %s

# REQUIRES: torch
# REQUIRES: kernel_bench

import subprocess
from pathlib import Path

tests = [
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": "32x32xf32xrnd,32x32xf32xid",
        "output_shape": "32x32xf32x0",
    },
    {
        "kernel": "level1/1_Square_matrix_multiplication_.py",
        "input_shapes": "32x32xbf16xrnd,32x32xbf16xid",
        "output_shape": "32x32xbf16x0",
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": "8x16xf32xrnd,16x8xf32xrnd",
        "output_shape": "8x8xf32x0",
    },
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": "8x16xbf16xrnd,16x8xbf16xrnd",
        "output_shape": "8x8xbf16x0",
    },
]

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent.parent
    kb_program = project_root / "tools" / "kernel_bench"
    kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"

    for test in tests:
        kb_kernel = kb_path / test["kernel"]
        command_line = [
            str(kb_program),
            str(kb_kernel),
            "--input-shapes",
            test["input_shapes"],
            "--output-shape",
            test["output_shape"],
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
# CHECK  0.37454012 0.9507143  0.7319939  ... 0.04645041 0.60754484 0.17052412
# CHECK: 0.27214515 0.59023064 0.3609739  ... 0.297349   0.9243962  0.97105825

# CHECK-NOT: Execution failed

# CHECK: 1_Square_matrix_multiplication_.mlir
# CHECK  0.375 0.949219 0.730469 ... 0.0463867 0.609375 0.170898
# CHECK: 0.271484 0.589844 0.361328 ... 0.296875 0.925781 0.972656

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 3.120935  3.7697    4.5365195 4.397648  4.4506536 3.2665431 3.5362916
# CHECK: 5.036752  5.312808  5.8109508 4.810084  4.7435184 4.35573   5.311559

# CHECK-NOT: Execution failed

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 3.125 3.76562 4.53125 4.40625 4.4375 3.26562 3.53125 3.9375
# CHECK: 5.03125 5.3125 5.8125 4.8125 4.75 4.34375 5.3125 5.5625

# CHECK-NOT: Execution failed

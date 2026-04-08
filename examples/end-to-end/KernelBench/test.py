#!/usr/bin/env python

# RUN: python %s 2>&1 | FileCheck %s
# CHECK-NOT: Execution failed

import subprocess
from pathlib import Path

if __name__ == "__main__":
    # This is a simple test to run the KernelBench example end-to-end.
    # It imports the PyTorch model, converts it to MLIR, runs the optimization pipeline, and executes the module.
    # The test passes if the output of the module matches the expected output from the PyTorch model.

    kb_program = Path(__file__).parent / "kernel_bench"
    project_root = Path(__file__).parent.parent.parent.parent
    kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"
    kb_kernels = [
        kb_path / kb_script
        for kb_script in [
            "level1/1_Square_matrix_multiplication_.py",
            "level1/2_Standard_matrix_multiplication_.py",
        ]
    ]
    initializers = [
        "32x32xf32xrnd,32x32xf32xrnd,32x32xf32x0",  # level1/1_Square_matrix_multiplication_.py
        "16x32xf32xrnd,32x16xf32xrnd,16x16xf32x0",  # level1/2_Standard_matrix_multiplication_.py
    ]

    for kb_kernel in kb_kernels:
        command_line = [
            str(kb_program),
            str(kb_kernel),
            "--input-shape",
            initializers[kb_kernels.index(kb_kernel)],
            "--print-tensor=3",
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

        assert result.returncode == 0, "Execution failed"

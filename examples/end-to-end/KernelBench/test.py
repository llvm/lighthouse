#!/usr/bin/env python

# RUN: python %s | FileCheck %s

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
        "32x32xf32x0,32x32xf32xrnd,32x32xf32xid",  # level1/1_Square_matrix_multiplication_.py
        "8x8xf32x0,8x16xf32xrnd,16x8xf32xrnd",  # level1/2_Standard_matrix_multiplication_.py
    ]

    for kb_kernel in kb_kernels:
        command_line = [
            str(kb_program),
            str(kb_kernel),
            "--input-shape",
            initializers[kb_kernels.index(kb_kernel)],
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

# CHECK: 2_Standard_matrix_multiplication_.mlir
# CHECK: 1.9275348  1.8850336  2.747824   1.5414746  0.64427626 2.0864286
# CHECK: 1.4014363  1.6915107  1.7900416  1.9984261  2.0468292  0.9830923

# CHECK-NOT: Execution failed

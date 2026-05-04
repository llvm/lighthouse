import subprocess
from pathlib import Path

tests = [
    {
        "kernel": "level1/2_Standard_matrix_multiplication_.py",
        "input_shapes": "256x64xf16x1,64x256xf16x1",
        "output_shape": "256x256xf16x0",
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
            "--target=xegpu",
            "--input-shapes",
            test["input_shapes"],
            "--output-shape",
            test["output_shape"],
            "--benchmark",
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

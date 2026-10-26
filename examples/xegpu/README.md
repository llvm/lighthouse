# XeGPU examples

## Installation

### 1. GPU Drivers and Level Zero

Install Intel GPU drivers and Level Zero runtime on your system.

### 2. Compile LLVM with Intel GPU support

To use Lighthouse with Intel GPUs, LLVM must be built with LevelZero runtime.

Set up a Python environment and install Python packages:

```bash
pip install pybind11 nanobind PyYAML numpy
```

Lighthouse requires a recent LLVM version. The default version, also tested by
our CI system, can be found in the [pyproject.toml](../../pyproject.toml) file.
For example, the following `mlir-python-bindings` entry indicates LLVM commit
`b6d7afe53`.

```toml
dependencies = [
    "mlir-python-bindings==20260330+b6d7afe53",
    ...
```

Set `LLVM_INSTALL_DIR` and use the below script to checkout and compile LLVM locally.

```bash
export LLVM_INSTALL_DIR=<...>
export LLVM_COMMIT=<...>

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $LLVM_COMMIT
mkdir -p build
cd build

cmake ../llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="SPIRV" \
  -DLLVM_INSTALL_GTEST=ON \
  -DMLIR_ENABLE_LEVELZERO_RUNNER=1 \
  -DMLIR_ENABLE_BINDINGS_PYTHON=1 \
  -DPython3_EXECUTABLE=$(which python3) \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
cmake --build .
cmake --install .
```

If cmake cannot find LevelZero, set environment variable `LEVEL_ZERO_DIR=<path-to-level-zero-install-root>`.

### Install Lighthouse

Install Lighthouse as instructed in the main [README](../../README.md).

Override the default LLVM package by setting `PYTHONPATH` to the local LLVM Python bindings:

```bash
export PYTHONPATH=${LLVM_INSTALL_DIR}/python_packages/mlir_core
```

If you wish to point to the build directory instead, do:

```bash
export PYTHONPATH=${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core/
```

## Matrix multiplication benchmark

Run the default 4k (float16, float16) -> float32 matrix-multiply-accumulate benchmark with correctness test:

```bash
python matmul.py --check-result
```

Set different M, N, K problem size

```bash
python matmul.py --sizes 1024 2048 4096 ...
```

To run matrix multiply (C = A * B) kernel instead of matrix-multiply-accumulate (C += A * B):

```bash
python matmul.py --no-accumulate-c ...
```

Run with bias and ReLU post-op:

```bash
python matmul.py --bias --relu ...
```

Set tiling parameters from the command line:

```bash
python matmul.py --wg-tile 128 256 ...
```

See all command-line arguments:

```bash
python matmul.py --help
```

## Multilayer Perceptron (MLP) benchmark

Run the default single layer MLP (batch=1024, input_features=1024, output_features=1024) benchmark with correctness test:

```bash
python mlp.py --check-result
```

which is equivalent to

```bash
python mlp.py -b 1024 -i 1024 -o 1024 --check-result
```

Run a 3-layer MLP with batch size 128:

```bash
python mlp.py -b 128 -i 16384 -o 8192 --hidden-sizes 16384 16384 ...
```

which corresponds to

```txt
MLP with 3 layers
  Layer 0: M=128, N=16384, K=16384
  Layer 1: M=128, N=16384, K=16384
  Layer 2: M=128, N=8192, K=16384
```

Add bias to all layers and ReLU to hidden layers:

```bash
python mlp.py --bias --relu ...
```

## Kernel tuning

### Matmul kernel tuning based on cost modeling

`tune_matmul_costmodel.py` uses cost-model estimates to select promising matrix multiplication configurations for benchmarking. Results are saved to `out_costmodel.csv`.

```bash
python tune_matmul_costmodel.py --sizes 1024 2048 4096 --dry-run
```

Omit `--dry-run` to run the tuning and benchmark selected configurations; use `--dump-json 5` to save the five best configurations.

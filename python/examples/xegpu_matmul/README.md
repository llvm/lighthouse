# XeGPU matrix multiplication benchmark

## Installation

### 1. GPU Drivers and Level Zero

Install Intel GPU drivers and Level Zero runtime on your system.

### 2. Compile LLVM with Intel GPU support

To use Lighthouse with Intel GPUs, LLVM must be built with LevelZero runtime.

Set up a Python environment and install Python packages:

```bash
pip install pybind11 nanobind PyYAML numpy
```

Set `LLVM_INSTALL_DIR` and use the below script to checkout and compile LLVM locally.

```bash
export LLVM_INSTALL_DIR=<...>
LLVM_VERSION=83765f435d1c
git checkout https://github.com/llvm/llvm-project.git -b $LLVM_VERSION

cd llvm-project
mkdir -p build
cd build

cmake ../llvm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
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

Install Lighthouse as instructed in the main [README](../../../README.md).

Override the default LLVM package by setting `PYTHONPATH` to the local LLVM Python bindings:

```bash
export PYTHONPATH=${LLVM_INSTALL_DIR}/python_packages/mlir_core
```

## Usage

Run the default 4k (float16, float16) -> float32 matrix multiplication benchmark with correctness test:

```bash
python matmul.py --check-result
```

Set different M, N, K problem size

```bash
python matmul.py --sizes 1024 2048 4096 ...
```

Run with ReLU post-op:

```bash
python matmul.py --relu ...
```

See all command line arguments:

```bash
python matmul.py --help
```

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

Set `LLVM_INSTALL_DIR` and use the below script to checkout and compile LLVM locally.

```bash
export LLVM_INSTALL_DIR=<...>
export LLVM_VERSION=45bee6efe9d6

git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout $LLVM_VERSION
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

### Exhaustive grid search

`tune_matmul_gridsearch.py` performs an exhaustive grid search on a matrix multiplication kernel. It takes similar arguments as the `matmul.py` benchmark:

```bash
python tune_matmul_gridsearch.py --sizes 1024 2048 4096 --bias --relu --no-accumulate-c
```

The executed parameter combinations are stored in `out_gridsearch.csv` file along with the obtained performance metrics:

```txt
m,n,k,wg_m,wg_n,sg_m,sg_n,k_tile,load_a_m,load_a_k,load_b_k,load_b_n,prefetch_a_m,prefetch_a_k,prefetch_b_k,prefetch_b_n,prefetch_nb,time (us),GFLOPS/s
4096,4096,4096,64,256,32,32,64,8,16,16,16,8,16,8,16,1,???,???
...
```

To get information about the search space (e.g., tile parameter choices) without actually executing the kernels run with `--dry-run` flag:

```bash
python tune_matmul_gridsearch.py --dry-run
```

Example output:

```txt
Matmul problem size: [4096, 4096, 4096]
ab_type='f16'
c_type='f32'
has_bias=False
has_relu=False
accumulate_c=True
Variable set:
wg_m=[64, 128, 256]
wg_n=[64, 128, 256]
sg_m=[32, 64, 128]
sg_n=[32, 64, 128]
k_tile=[16, 32, 64, 128, 256]
load_a_m=[8, 16, 32]
load_a_k=[8, 16, 32]
load_b_k=[8, 16, 32]
load_b_n=[8, 16, 32]
prefetch_a_m=[8, 16, 32]
prefetch_a_k=[8, 16, 32]
prefetch_b_k=[8, 16, 32]
prefetch_b_n=[8, 16, 32]
prefetch_nb=[1]
Total complexity: 2657205 configurations
Number of executed configurations: 5292
```

Total complexity is the number of parameter combinations without any filtering. The number of executed configurations shows the number of valid combinations, i.e. ones that satisfy appropriate (e.g., hardware) constraints.

To dump the best found configurations as JSON files at the end of the search, use `--dump-json n` flag where `n` stands for the number of best configurations. The files are named as `matmul_params_*_00.json` with increasing integer suffix (best configuration being 00).

> [!NOTE]
> Running the grid search typically takes several hours to complete.

### Adaptive sampling with Genetic Algorithm

`tune_matmul_ga.py` employs a genetic algorithm for adaptive sampling to explore the kernel tuning search space. This approach is typically an order of magnitude faster while discovering high throughput parameter combinations.

The command-line interface is similar to `tune_matmul_gridsearch.py`:

```bash
python tune_matmul_ga.py --sizes 1024 2048 4096 --bias --relu --dump-json 10
```

The executed parameter combinations are stored in `out_genetic_algorithm.csv` file.

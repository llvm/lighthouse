# XeGPU Multilayer Perceptron (MLP) benchmark

## Installation

To install Lighthouse with XeGPU support, see installation instructions in [xegpu_matmul/README.md](../xegpu_matmul/README.md).

## Usage

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

Add ReLU to all layers:

```bash
python mlp.py --relu ...
```

See all command line arguments:

```bash
python mlp.py --help
```

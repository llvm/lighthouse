# Lighthouse Ingress

The `lighthouse.ingress` module converts various input formats to MLIR modules.

## Supported Formats

### Torch
Convert PyTorch models to MLIR using `lighthouse.ingress.torch`.

**Examples:** [torch examples](https://github.com/llvm/lighthouse/tree/main/examples/ingress/torch)

### mlir_gen
Generate models from the CLI or through a small library of Python helpers: `lighthouse.ingress.mlir_gen`.

**Examples:** [mlir_gen examples](https://github.com/llvm/lighthouse/tree/main/examples/ingress/mlir_gen)

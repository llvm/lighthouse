# Lighthouse Tools

## lh-opt

This tool helps test and validate assumptions of the Lighthouse classes for building compilers.

For now, the Driver will register some passes that you can pass as command line argument.
In the future we should enable user-registration of passes, bundles, schedules, etc.

The way to use it is demonstrated in the test `test/opt/pipeline-check.mlir`:

```
// Runs one pass over the file
lh-opt --stage=canonicalize file.mlir

// Runs one pass bundle over the file
lh-opt --stage=BufferizationBundle file.mlir

// Runs one transform over the file
lh-opt --stage=my-transform.mlir file.mlir

// Runs a whole pipeline on the file
lh-opt --stage=BufferizationBundle --stage=canonicalize --stage=my-transform.mlir --stage=canonicalize --stage=LLVMLoweringBundle file.mlir
```

Note, this basic functionality is for testing purposes.

## lh-run

Executes a payload MLIR file, with optional optimizing pipeline.

The way to use is demonstrated in the test `test/run/pipeline-check.mlir`:

```
// Runs module with random inputs
lh-run --stage=my-pipeline.yaml file.mlir --entry-point=entry --input-shape=256x512,512x1024 --input-type=f32

// Benchmarks the module above with random inputs
lh-run --stage=my-pipeline.yaml file.mlir --entry-point=entry --input-shape=256x512,512x1024 --input-type=f32 --benchmark
```

Note, this basic functionality is for testing purposes.

## kernel-bench

End-to-end tool that takes in a [KernelBench](https://github.com/ScalingIntelligence/KernelBench) program, input and output shapes, and executes it. This is an example of how you would build a compiler using Lighthouse's modules.

The way to use is demonstrated in the test `examples/end-to-end/KernelBench/test_kernel_bench.py`:

```
// Runs level 1 / kernel 1 on small `f32` shapes and prints the output
kernel-bench level1/1_Square_matrix_multiplication_.py --input-shapes 32x32xf32xrnd,32x32xf32xid --output-shapes 32x32xf32x0 --print-tensor 1

// Same as above, but printing the IR after every compiler stage
kernel-bench level1/1_Square_matrix_multiplication_.py --input-shapes 32x32xf32xrnd,32x32xf32xid --output-shapes 32x32xf32x0 --print-mlir-after-all

// Benchmarks level 1 / kernel 2 on large `bf16` shapes
kernel-bench level1/2_Standard_matrix_multiplication_.py --input-shapes 1024x2048xbf16xrnd,2048x4096xbf16xrnd --output-shapes 1024x4096xbf16x0 --benchmark
```

Uses the in-tree submodule of KernelBench.

Note: input/output syntax is experimental. See the `execution/init` module for more details.

## lh-tune

TODO: Write short doc and examples on how to use

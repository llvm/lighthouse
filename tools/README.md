# Lighthouse Tools

## lh-opt

This tool helps test and validate assumptions of the Lighthouse classes for building compilers.

For now, the Driver will register some passes that you can pass as command line argument.
In the future we should enable user-registration of pases, bundles, schedules, etc.

The way to use is demonstrated in the test `test/opt/pipeline-check.mlir`:

```
// Runs one pass over the file
lh-opt --stage=canonicalize file.mlir

// Runs one pass bundle over the file
lh-opt --stage=BufferizationBundle file.mlir

// Runs one trasnform over the file
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

## lh-tune

TODO: Write short doc and examples on how to use

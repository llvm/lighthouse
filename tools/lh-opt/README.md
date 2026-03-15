# Lighthouse Opt Tool

This tool helps test and validate assumptions of the Lighthouse classes for building compilers.

For now, the Driver will register some passes that you can pass as command line argument.
In the future we should enable user-registration of pases, bundles, schedules, etc.

The way to use is demonstrated in the test `test/opt/pipeline-check.mlir`:

```
// Runs one pass bundle over the file
lh-opt --stage=bufferize file.mlir

// Runs multiple bundles in order on the file
lh-opt --stage=bufferize --stage=mlir_lowering --stage=llvm_lowering file.mlir
```

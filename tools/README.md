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

Note, this basic functionality is for testing purposes. For building larger pipelines a new method will need to be created.
One idea is to use structured text files, like YAML or JSON, and then use `--stage=my-pipeline.json`.

## lh-tune

TODO: Write short doc and examples on how to use

# Kernel Bench Testing & Benchmarking

This example drives the tool `kernel-bench` to load, compile, run and benchmark Kernel Bench kernels.

Please review the [kernel-bench](../../../tools/README.md#kernel-bench)'s own documentation, for more information on its usage.

## Overview

There are three core components of this example:
1. The lists of kernels, arguments, performance, etc (`level1.yaml`, `level2.yaml`).
2. A `schedules` directory with descriptor files that are used with the kernels above.
3. The `test-kernel-bench` tool that uses the kernel arguments and the schedules to run kernels.

Example:
```bash
# Runs kernel 2 from level 1 and compares the output with PyTorch eager for correctness
$ uv run examples/KernelBench/test-kernel-bench.py --kernel level1/2_

STDOUT:
Executing the module...
Success: The output of the compiled model matches the reference output.
Tolerance: 0.01

STDERR:

Return code: 0
```

In the example above, `test-kernel-bench` will parse `level1.yaml`, take the arguments for kernel `2_...` and call `kernel-bench` with the appropriate arguments.

Any argument that is unknown to `test-kernel-bench` will be passed as-is to `kernel-bench`, so you can fine tune the execution without having to copy-and-paste command lines over.

Note that the `--kernel` option uses `startsWith`, so if you provide `level1/2` instead, it will execute kernels `2_...`, `20_...`, `21_...` etc.
That's why the invocation above has the _underscore_ after the `2` (`2_`), so that only one is executed.

## Execution Modes

The `kernel-bench` tool has two execution modes: **Import** and **Torch Compile**.

### Import Mode

This mode will use Torch MLIR to import the MLIR module into a separate file, then execute that file.
This is helpful when investigating the IR, injecting that IR into other runners, modifying the IR by hand, etc.

Since the IR generated has a return value (the function `forward` does), we need some adjusting to execute the function as a `main` function.
Basically, `kernel-bench` runs some passes to convert return values to arguments and bufferizes it back into it.
This is done automatically, regardless of the schedule you select to pass, and you don't need to do anything.

Input and output buffers are automatically allocated in Python and linked to the Module's execution.
This is how we pass arguments and capture the return values.
This is a _major_ problem in this mode, since some tensors are huge and can exhaust the host's memory.

This is the _default_ mode, and will be used in the command line above.
For now, it's the only mode that supports benchmarking (option `--benchmark`).

### Torch Compile Mode

This mode will _compile_ the module using a Torch compiler backend that calls our MLIR importer, runs our schedules and returns the optimized MLIR module.
The logic is supposed to be identical to the mode above, so any deviations should be treated as a bug.

The main differences between this mode and the import mode are:
* The IR received by the optimizer is _whatever subgraph_ PyTorch decides to give us, which may be more than one. Each will be optimized separately.
* There is no need to allocate extra tensors or handle return values, since PyTorch does all that for us.
* We do not have an independent execution, so we can't yet run benchmarks with it.

This is the _expected_ mode that people will use in production.
Once we can run benchmarks and have fixed all the bugs in this mode, it will become the _default mode_.

## Key Details

### Core Arguments

The key arguments in `test-kernel-bench` are:
* `--kernel`: Specifies the kernel's `startsWith` name. Use to run one or more kernels.
* `--benchmark`: Selects to run benchmarks (import mode only for now), and dispatches 100 iterations.
  The entry in `levelN.yaml` file must have the `gflops` set to calculate performance.
* `--dtype`: Selects the data type for the kernel. Some kernels only support f32. Default is f32.
* `--infer-shapes`: Uses the Kernel Bench module's own functions to infer input/output shapes.
  Default is to read from the YAML file. Some kernels are too big and the YAML has smaller sizes override.
* `--target` and `--feature`: Decides which schedule sub-class to search schedules from.
  Featues are looked up first (ex. `avx512`) and if not found, the target (ex. `x86_64`).

### YAML Syntax

```yaml
- kernel: level1/1_Square_matrix_multiplication_.py
  input_shapes: [1024x1024, 1024x1024]
  initializations: [rnd, id]
  output_shape: 1024x1024
  init_args: []
  gflops: (1024 * 1024 * 1024 * 2) / 1e9
  pipeline: matmul
```

* `kernel`: Level and kernel name, to search inside Kernel Bench's submodule.
* `input_shapes` and `output_shapes`: Dimensions and sizes for input/output shapes.
* `init_args`: Arguments to initialize the module with (before compilation).
* `initializations`: What type to initialize the inputs (`rnd`, `id`, `0`)
* `gflops`: Calculation on how to get the number of floating point operations in this kernel (for performance measurement).
* `pipeline`: Name of the sub-dir, replaces: `schedules/$target/$pipeline/$dtype.yaml`.

### Schedules

The descriptor schedules (YAML format) are stored in the root `schedules` directory.

They are organized in the following way:
* Target agnostic schedules live in the root directory.
* Target specific schedules, but not kernel specific, live in the target directory (`schedules/$target`).
* Kernel specific schedules live inside a kernel type name (ex. `matmul`) inside a particular target.
  Different targets would have different schedules (parameters) for the same logic anyway.
* Inside the target directory, there are multiple YAML files, one per datatype (ex. `f32.yaml`, `bf16.yaml`, etc).

Descriptor files can _include_ other files, and that works with relative paths, so you'll see the following in some pipelines:
```yaml
  - include: ../lower.yaml
```

So the `x86_64/matmul/f32.yaml` file will include the `x86_64/lower.yaml` with that directive.

_Note: In `kernel-bench`, you can **always** override the pipeline with the `--pipeline` option._

## Pipeline Development

The main reason for this example is to be able to quickly iterate through each kernel and fine-tune parameters and schedules for a particular target.

A reasonable cycle would be:
1. Choose a kernel that doesn't run fast with its current schedule (or uses the default schedule).
2. Create a name for the new schedule (for example, `elementwise`) and name it in the YAML file's `pipeline` key.
3. Create that directory inside your target, and in there an `f32.yaml` file, perhaps copying from another similar pipeline.
4. Run `test-kernel-bench` on that kernel, which will pick up the new pipeline and the correct arguments, and using the `--benchmark` flag, inspect the performance.
5. Fine tune the pipeline YAML file, add new passes, transforms, change parameters, and run the benchmark again, and again.
6. Once happy with the performance, commit the new pipeline and changes, and submit a new PR.

This also works when adding new targets.
Copy and paste from an existing target, and adapt the schedules to your new target.
Then benchmark each kernel on each data type to validate your choices.
Submit a PR with the new target.

To reduce the load on reviewers, please submit PRs with small changes, and iteratively go through them upstream, rather than doing all the work downstream and then submitting a massive PR with all changes.
This also reduces your work downstream while you're working, minimizing rebase work as the upstream tree evolves.

## Usage Examples

### Smoke Tests

Runs all registered kernels with the default schedule (lower-to-loops) on `Torch Compile` mode:
```bash
$ uv run examples/KernelBench/test-kernel-bench.py --smoke-test --torch-compile
```
_Note: This takes a loooong time..._

### Benchmark

Benchmarks a kernel with its default options
```bash
$ uv run examples/KernelBench/test-kernel-bench.py --kernel level1/1_ --benchmark

Running command: /home/rengolin/devel/llvm/lighthouse/tools/kernel-bench /home/rengolin/devel/llvm/lighthouse/third_party/KernelBench/KernelBench/level1/1_Square_matrix_multiplication_.py --pipeline /home/rengolin/devel/llvm/lighthouse/examples/KernelBench/schedules/x86_64/matmul/f32.yaml --benchmark --input-shapes 1024x1024xf32xrnd,1024x1024xf32xid --output-shape 1024x1024xf32x0 --init-args None

STDOUT:
Running the benchmark...
100 runs: 0.015572094917297363 seconds
Success: The output of the compiled model matches the reference output.
Tolerance: 0.01

Performance: 137.91 GFLOPS

STDERR:

Return code: 0
```
_Note: This will show the `kernel-bench` command line, verify the correctness of the output with PyTorch eager, show the mean time of 100 runs and calculate the performance if the YAML file has the `gflops` key set._

### Compare MLIR files

To compare the MLIR generated between Import and Compile modes, call both with the `-print-original-module` argument:
```bash
$ uv run examples/KernelBench/test-kernel-bench.py --kernel level2/11_ --print-original-module > import.mlir

$ uv run examples/KernelBench/test-kernel-bench.py --kernel level2/11_ --print-original-module --torch-compile > compile.mlir
```
_Note: there will be some verbosity in the text file, before and after the MLIR module._

### Debug the schedule

To debug the schedule you selected, you can use the `--print-mlir-after-all` option and see the evolution of your IR as you pass through your pipeline.
If the pipeline crashes, you'll see exactly where it crashed and have the full context.

This isn't a `test-kernel-bench` options, and is an example of forwarding the arguments into `kernel-bench` directly.

```bash
$ uv run examples/KernelBench/test-kernel-bench.py --kernel=level2/11 --print-mlir-after-all
```
_Note: You can also run this between import and compile modes to see the difference on their own evolution through the same pipeline._

### Dump assembly

To dump the assembly generated by the `ExecutionEngine` into a `.s` file, run the program with:

```bash
$ uv run examples/KernelBench/test-kernel-bench.py --kernel level1/1_ --dump-assembly
...
Assembly written to 1_Square_matrix_multiplication_.s
```

The file will be at the directory where you called the script, regardless of where the PyTorch model file was.

_Note: This simply calls `objdump` on the object file dumped by the `ExecutionEngine`, so YMMV on the output._

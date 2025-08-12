Main driver for the MLIR schedule builder and runner.

This script is divided in three parts:
 1. Using the schedule directory, builds schedules based on user input.
 2. Runs the generated schedules through `mlir-opt` and dumps the intermediate IR.
 3. (Optional) Runs the output through `mlir-runner` and dumps the results.

The `opt` and `runner` commands can be customized through the command line arguments.
This allows tests to run on different downstream projects, using their own pipelines and arguments.

The ingress process should output files into its `cache` directory, which is easy for the harness to access.
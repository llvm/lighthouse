#! /usr/bin/env python3
# Main driver for the MLIR schedule builder and runner.
#
# This script is divided in three parts:
#  1. Using the schedule directory, builds schedules based on user input.
#  2. Runs the generated schedules through `mlir-opt` and dumps the intermediate IR.
#  3. (Optional) Runs the output through `mlir-runner` and dumps the results.
#
# The `opt` and `runner` commands can be customized through the command line arguments.
# This allows tests to run on different downstream projects, using their own pipelines and arguments.
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MLIR Schedule Driver")
    parser.add_argument("--opt", type=str, default="mlir-opt", help="Optimizer tool (defaults to mlir-opt)")
    parser.add_argument("--opt-args", type=str, default="", help="Arguments for mlir-opt")
    parser.add_argument("--run", action="store_true", help="Whether to run the generated IR")
    parser.add_argument("--runner", type=str, default="mlir-runner", help="Runner tool (defaults to mlir-runner)")
    parser.add_argument("--runner-args", type=str, default="", help="Arguments for mlir-runner")
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    print(f"Using optimizer: {args.opt} with arguments: {args.opt_args}")
    if args.run:
        print(f"Running with runner: {args.runner} and arguments: {args.runner_args}")
    # TODO: Implement
    pass

# Init
if __name__ == "__main__":
    main()
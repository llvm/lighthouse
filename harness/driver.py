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
import os
import sys

from Logger import Logger
from Execute import Execute

def parse_args():
    parser = argparse.ArgumentParser(description="MLIR Schedule Driver")
    # Base arguments
    parser.add_argument(
        "-i", "--input", type=str, default="", help="Input file (defaults to stdin)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="", help="Output file (defaults to stdout)"
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite the output if it exists"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="The verbosity of logging output",
    )

    # Optimization pipeline
    parser.add_argument(
        "--opt",
        type=str,
        default="mlir-opt",
        help="Optimizer tool (defaults to mlir-opt)",
    )
    parser.add_argument(
        "--opt-args", type=str, default="", help="Arguments for mlir-opt"
    )

    # Call runner on the output of mlir-opt. Must be in LLVM dialect
    parser.add_argument(
        "-r", "--run", action="store_true", help="Whether to run the generated IR"
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="mlir-runner",
        help="Runner tool (defaults to mlir-runner)",
    )
    parser.add_argument(
        "--runner-args", type=str, default="", help="Arguments for mlir-runner"
    )

    # Benchmarking (TODO: add infra to create wrapper loop)
    parser.add_argument(
        "-b", "--benchmark", action="store_true", help="Enable benchmarking"
    )
    parser.add_argument(
        "--benchmark-loops", type=int, default=10, help="Number of benchmark loops"
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=10,
        help="Number of benchmark warmup iterations",
    )

    # System level options
    parser.add_argument(
        "-S", "--schedule-path", type=str, default="", help="Schedule library directory"
    )
    parser.add_argument(
        "-L",
        "--library-path",
        type=str,
        default="",
        help="Runtime library path (shared objects)",
    )

    # For target machine, data layout
    parser.add_argument("--target", type=str, default="", help="Target architecture")
    parser.add_argument(
        "--chip", type=str, default="", help="Target chip name (CPU, GPU)"
    )
    parser.add_argument(
        "--target-features",
        type=str,
        default="",
        help="Target chip features (AVX, AVX2, etc.)",
    )

    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    logger = Logger("driver", args.verbose)

    # Validate Input/Output arguments
    args.input = os.path.abspath(args.input) if args.input else "-"
    if args.input and args.input != "-" and not os.path.isfile(args.input):
        logger.error(f"Invalid input file: {args.input}")
        exit(1)
    logger.info(f"Input file: {args.input}")
    args.output = os.path.abspath(args.output) if args.output else "-"
    if (
        not args.force
        and args.output
        and args.output != "-"
        and os.path.isfile(args.output)
    ):
        logger.error(f"Output file already exists: {args.output}")
        exit(1)

    logger.info(f"Output file: {args.output}")
    logger.info(f"Force: {args.force}")

    # Check arguments
    logger.info(f"Using optimizer: {args.opt} {args.opt_args}")
    if args.run:
        logger.info(f"Using runner: {args.runner} {args.runner_args}")
    if args.benchmark:
        logger.info(
            f"Benchmarking enabled with {args.benchmark_loops} loops and {args.benchmark_warmup} warmup iterations"
        )
    if args.schedule_path:
        logger.info(f"Using schedule path: {args.schedule_path}")
    if args.library_path:
        logger.info(f"Using library path: {args.library_path}")
    logger.info(f"Using target: ({args.target}, {args.chip}, {args.target_features})")
    # TODO: Implement
    pass

# Init
if __name__ == "__main__":
    main()

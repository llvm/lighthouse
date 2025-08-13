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
import tempfile

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
        "-s", "--schedule", type=str, default="", help="Schedule file to apply to the input (must exist)"
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

# Validate command line arguments
def validate_args(args, logger):
    # Validate Input/Output arguments
    args.schedule = os.path.abspath(args.schedule)
    if not os.path.isfile(args.schedule):
        logger.error(f"Invalid (mandatory) schedule file: '{args.schedule}'")
        exit(1)
    logger.info(f"Schedule file: {args.schedule}")

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
    pass

# Prepare the schedule based on the pipeline schedule and the library directory
def prepare_schedule(args, logger):
    # Load the pipeline schedule
    with open(args.schedule, "r") as f:
        schedule = f.read()

    # TODO: Include other schedules as needed from the library and produce a textual version of the schedule to be included in the payload IR

    return schedule

# Run the optimization
def optimize_schedule(schedule, args, logger):
    # Prepend the schedule to the payload IR and save to a temporary
    temp = tempfile.NamedTemporaryFile(delete=True)
    payload_ir = f"// Schedule\n{schedule}\n\n// Payload IR\n"
    with open(args.input, "r") as f:
        payload_ir += f.read()
    temp.write(payload_ir.encode())
    temp.flush()

    # Run optimizer (TODO: )
    command = [args.opt, temp.name]
    command.extend(args.opt_args.split())
    res = Execute(logger).run(command)

    return res, temp

# Run the generated IR
def run_generated_ir(file, args, logger):
    # Prepare the command
    command = [args.runner, file.name]
    command.extend(args.runner_args.split())
    res = Execute(logger).run(command)

    return res

# Benchmark the generated IR
def benchmark_generated_ir(file, args, logger):
    for i in range(args.benchmark_loops):
        logger.info(f"Running benchmark iteration {i + 1}")
        res = run_generated_ir(file, args, logger)
        logger.info(f"Benchmark iteration {i + 1} result: {res}")

    return res

# Main function
def main():
    args = parse_args()
    logger = Logger("driver", args.verbose)
    validate_args(args, logger)

    # Prepare the schedule
    schedule = prepare_schedule(args, logger)

    # Run the optimization
    res, file = optimize_schedule(schedule, args, logger)

    # Run the generated IR
    if args.run:
        res = run_generated_ir(file, args, logger)

    # Benchmark the generated IR
    if args.benchmark:
        res = benchmark_generated_ir(file, args, logger)

# Init
if __name__ == "__main__":
    main()

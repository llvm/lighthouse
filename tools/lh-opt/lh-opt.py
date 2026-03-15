#! /usr/bin/env python

import argparse
import os

from mlir import ir
from lighthouse.pipeline.opt import Driver


def import_payload(path: str) -> ir.Module:
    """Import an MLIR text file into the payload module"""
    if path is None:
        raise ValueError("Path to the payload module must be provided.")
    if not os.path.exists(path):
        raise ValueError(f"Path to the payload module does not exist: {path}")
    with ir.Context():
        with open(path, "r") as f:
            return ir.Module.parse(f.read())


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(
        description="""
    Lighthouse Optimization Pipeline: Applies a series of transformations to an input MLIR module,
    and produces an optimized MLIR module as output. The transformations are applied in argument order.
    The names of the passes are registered by the driver.
    """
    )
    Parser.add_argument(
        "payload_module", type=str, help="Path to the payload MLIR module to optimize."
    )
    Parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="List of transformations to apply to the input module.",
    )
    args = Parser.parse_args()

    # Load the input module.
    input_module = import_payload(args.payload_module)

    # Create the driver and run the transformations.
    driver = Driver(input_module, stages=args.stage)

    # Run the pipeline and get the optimized module.
    optimized_module = driver.run()
    print(optimized_module)

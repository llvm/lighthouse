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


def create_driver(module: ir.Module, stages: list[str]) -> Driver:
    """Create the driver's pipeline by selecting the passes that will run"""
    driver = Driver(module)
    if not stages:
        # Add all stages if no stages are specified.
        driver.bufferize()
        driver.mlir_lowering()
        driver.llvm_lowering()
        driver.cleanup()
    else:
        for t in stages:
            if t == "bufferize":
                driver.bufferize()
            elif t == "mlir_lowering":
                driver.mlir_lowering()
            elif t == "llvm_lowering":
                driver.llvm_lowering()
            elif t == "cleanup":
                driver.cleanup()
            else:
                raise ValueError(f"Unsupported transformation: {t}")

    return driver


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description="Lighthouse Optimization Pipeline")
    Parser.add_argument(
        "payload_module", type=str, help="Path to the payload MLIR module to optimize."
    )
    Parser.add_argument(
        "--stage",
        action="append",
        help="List of transformations to apply to the input module. Supported transformations: bufferize, lower_to_llvm, mlir_lowering, cleanup.",
    )
    args = Parser.parse_args()

    # Load the input module.
    input_module = import_payload(args.payload_module)

    # Create the driver and run the transformations.
    driver = create_driver(input_module, args.stage)

    # Run the pipeline and get the optimized module.
    optimized_module = driver.run()
    print(optimized_module)

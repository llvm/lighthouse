#!/usr/bin/env python3

import argparse
import os
import torch
import torch.nn as nn
from torch_mlir import fx
from torch_mlir.fx import OutputType

# Parse arguments for selecting which model to load and which MLIR dialect to generate
def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLIR for Torch-MLIR models.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the Torch model file.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        choices=["torch", "linalg", "stablehlo", "tosa"],
        default="linalg",
        help="MLIR dialect to generate.",
    )
    return parser.parse_args()

# Functin to load the Torch model
def load_torch_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    model = torch.load(model_path)
    return model

# Function to generate MLIR from the Torch model
# See: https://github.com/MrSidims/PytorchExplorer/blob/main/backend/server.py#L237
def generate_mlir(model, dialect):

    # Convert the Torch model to MLIR
    output_type = None
    if dialect == "torch":
        output_type = OutputType.TORCH
    elif dialect == "linalg":
        output_type = OutputType.LINALG
    elif dialect == "stablehlo":
        output_type = OutputType.STABLEHLO
    elif dialect == "tosa":
        output_type = OutputType.TOSA
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")

    module = fx.export_and_import(model, "", output_type=output_type)
    return module

# Main function to execute the script
def main():
    args = parse_args()
    
    # Load the Torch model
    model = load_torch_model(args.model)
    
    # Generate MLIR from the model
    mlir_module = generate_mlir(model, args.dialect)
    
    # Print or save the MLIR module
    print(mlir_module)

# Entry point for the script
if __name__ == "__main__":
    main()
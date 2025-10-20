"""
Example demonstrating how to load an already instantiated PyTorch model
to MLIR using Lighthouse.

The script uses 'lighthouse.ingress.torch.import_from_model' function that
takes an already instantiated PyTorch model, along with its sample inputs.
The function passes the model to torch_mlir to get a MLIR module in the
specified dialect.

The script uses model from 'DummyMLP/model.py' as an example.
"""

import torch

# MLIR infrastructure imports (only needed if you want to manipulate the MLIR module)
import mlir.dialects.func as func
from mlir import ir, passmanager

# Lighthouse imports
from lighthouse.ingress.torch import import_from_model

# Import a sample model definition
from .DummyMLP.model import DummyMLP

# Step 1: Instantiate a model and prepare sample input
model = DummyMLP()
sample_input = torch.randn(1, 10)

ir_context = ir.Context()
# Step 2: Convert PyTorch model to MLIR
mlir_module_ir : ir.Module = import_from_model(
    model,
    sample_args=(sample_input,),
    ir_context=ir_context
)

# Step 3: Extract the main function operation from the MLIR module and print its metadata
func_op : func.FuncOp = mlir_module_ir.operation.regions[0].blocks[0].operations[0]
print(f"entry-point name: {func_op.name}")
print(f"entry-point type: {func_op.type}")

# Step 4: Apply some MLIR passes using a PassManager (optional)
pm = passmanager.PassManager(context=ir_context)
pm.add("one-shot-bufferize")
pm.add("linalg-specialize-generic-ops")
pm.run(mlir_module_ir.operation)

# Step 5: Output the final MLIR
print("\n\nModule dump after running pm.Pipeline:")
mlir_module_ir.dump()

"""
Example demonstrating how to load a PyTorch model to MLIR using Lighthouse
without instantiating the model on the user's side.

The script uses 'lighthouse.ingress.torch.import_from_file' function that
takes a path to a Python file containing the model definition, along with
the names of functions to get model init arguments and sample inputs. The function
imports the model class on its own, instantiates it, and passes it to torch_mlir
to get a MLIR module in the specified dialect.

The script uses the model from 'DummyMLP/model.py' as an example.
"""

import os
from pathlib import Path

# MLIR infrastructure imports (only needed if you want to manipulate the MLIR module)
import mlir.dialects.func as func
from mlir import ir, passmanager

# Lighthouse imports
from lighthouse.ingress.torch import import_from_file

# Step 1: Set up paths to locate the model definition file
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
model_path = script_dir / "DummyMLP" / "model.py"

ir_context = ir.Context()

# Step 2: Convert PyTorch model to MLIR
# Conversion step where Lighthouse:
# - Loads the DummyMLP class and instantiates it with arguments obtained from 'get_init_inputs()'
# - Calls get_sample_inputs() to get sample input tensors for shape inference
# - Converts PyTorch model to linalg-on-tensors dialect operations using torch_mlir
mlir_module_ir : ir.Module = import_from_file(
    model_path,                              # Path to the Python file containing the model
    model_class_name="DummyMLP",             # Name of the PyTorch nn.Module class to convert
    init_args_fn_name="get_init_inputs",     # Function that returns args for model.__init__()
    inputs_args_fn_name="get_sample_inputs", # Function that returns sample inputs to pass to 'model(...)'
    dialect="linalg-on-tensors",             # Target MLIR dialect (linalg ops on tensor types)
    ir_context=ir_context                    # MLIR context for the conversion
)

# The PyTorch model is now converted to MLIR at this point. You can now convert
# the MLIR module to a text form (e.g. 'str(mlir_module_ir)') and save it to a file.
#
# The following optional MLIR-processing steps are to give you an idea of what can
# also be done with the MLIR module.

# Step 3: Extract the main function operation from the MLIR module and print its metadata
func_op : func.FuncOp = mlir_module_ir.operation.regions[0].blocks[0].operations[0]
print(f"entry-point name: {func_op.name}")
print(f"entry-point type: {func_op.type}")

# Step 4: Apply some MLIR passes using a PassManager
pm = passmanager.PassManager(context=ir_context)
pm.add("linalg-specialize-generic-ops")
pm.add("one-shot-bufferize")
pm.run(mlir_module_ir.operation)

# Step 5: Output the final MLIR
print("\n\nModule dump after running the pipeline:")
mlir_module_ir.dump()

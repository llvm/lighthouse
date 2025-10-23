"""
Example demonstrating how to load an already initialized PyTorch model
to MLIR using Lighthouse.

The script uses the 'lighthouse.ingress.torch.import_from_model' function that
takes an initialized PyTorch model (an instance of a Python class derived from 'nn.Module'),
along with its sample inputs. The function passes the model to torch_mlir
to get a MLIR module in the specified dialect.

The script uses a model from 'MLPModel/model.py' as an example.
"""

import torch

# MLIR infrastructure imports (only needed if you want to manipulate the MLIR module)
import mlir.dialects.func as func
from mlir import ir

# Lighthouse imports
from lighthouse.ingress.torch import import_from_model

# Import a sample model definition
from MLPModel.model import MLPModel

# Step 1: Instantiate a model class and prepare sample input
model = MLPModel()
sample_input = torch.randn(1, 10)

ir_context = ir.Context()
# Step 2: Convert the PyTorch model to MLIR
mlir_module_ir: ir.Module = import_from_model(
    model,
    sample_args=(sample_input,),
    ir_context=ir_context
)

# The PyTorch model is now converted to MLIR at this point. You can now convert
# the MLIR module to a text form (e.g. 'str(mlir_module_ir)') and save it to a file.
#
# The following optional MLIR-processing steps are to give you an idea of what can
# also be done with the MLIR module.

# Step 3: Extract the main function operation from the MLIR module and print its metadata
func_op: func.FuncOp = mlir_module_ir.operation.regions[0].blocks[0].operations[0]
print(f"entry-point name: {func_op.name}")
print(f"entry-point type: {func_op.type}")

# Step 4: output the imported MLIR module
print("\n\nModule dump:")
mlir_module_ir.dump()

# You can alternatively write the MLIR module to a file:
# with open("output.mlir", "w") as f:
#     f.write(str(mlir_module_ir))

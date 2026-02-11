# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: torch

import torch
import torch.nn as nn
from torch_mlir.fx import OutputType

from mlir import ir
from mlir.passmanager import PassManager

from lighthouse.ingress.torch import cpu_backend


def lower_to_llvm(module: ir.Module) -> ir.Module:
    """
    Lower MLIR ops within the module to MLIR LLVM IR dialect.

    A PyTorch model is expected to be imported as Linalg ops using
    tensor abstraction.
    This compilation function applies simple rewrites (notably bufferization
    and scalar lowering of the compute) in preparation for further jitting.

    Args:
        module: MLIR module coming from PyTorch importer.
    Returns:
        ir.Module: MLIR module with lowered IR
    """
    pm = PassManager("builtin.module", module.context)

    # Preprocess.
    # Use standard C interface wrappers for functions.
    pm.add("func.func(llvm-request-c-wrappers)")

    # Bufferize.
    pm.add(
        "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}"
    )
    pm.add("drop-equivalent-buffer-results")
    pm.add("buffer-deallocation-pipeline")
    pm.add("cse")
    pm.add("canonicalize")

    # Lower to LLVM.
    pm.add("convert-linalg-to-loops")
    pm.add("convert-scf-to-cf")
    pm.add("convert-to-llvm")
    pm.add("reconcile-unrealized-casts")

    # Cleanup
    pm.add("cse")
    pm.add("canonicalize")

    # IR is transformed in-place.
    pm.run(module.operation)

    # Return the same module which now holds LLVM IR dialect ops.
    return module


def jit_compile_model_decorator():
    # The MLIR backend is provided as a callback to the PyTorch compile function.
    #
    # When the model is invoked, a traced graph is given into the custom backend
    # which imports the PyTorch model into MLIR IR, uses the provided compilation
    # function, and finally jits the model into an executable function.
    @torch.compile(
        dynamic=False,
        backend=cpu_backend(lower_to_llvm, dialect=OutputType.LINALG_ON_TENSORS),
    )
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.matmul(a, b)

    # Compute reference result.
    a = torch.randn(8, 16)
    b = torch.randn(16, 4)
    out_ref = torch.matmul(a, b)

    # Create a model instance as usual.
    model = Model()

    # Call jitted function directly with PyTorch tensors.
    #
    # At the time of invocation, the PyTorch model is traced and always specialized
    # into a graph with static shapes, based on the inputs, as the dynamic shape tracing
    # is disabled (dynamic=False).
    #
    # Then the MLIR backend is triggered and the provided PyTorch graph is converted
    # into an MLIR module. The imported IR is then jitted into an executable function.
    #
    # Note that due to this JIT process, the first call for any given set of inputs
    # incurs compilation overhead cost. Already jitted functions are cached and managed
    # by torch.compile infrastructure which can speed up subsequent calls.
    #
    # Inputs are implicitly converted to packed C-type argument before invoking
    # jitted MLIR function.
    # The backend also manages model's outputs and returns standard PyTorch tensors
    # as per torch.compile contract.
    out = model(a, b)
    is_match = torch.allclose(out_ref, out, rtol=0.001, atol=0.001)

    # CHECK: Input 1 - result match: True
    print(f"Input 1 - result match: {is_match}")

    # Change input shapes and recompute reference result.
    a = torch.randn(2, 4)
    b = torch.randn(4, 16)
    out_ref = torch.matmul(a, b)

    # Compile another specialized model.
    out = model(a, b)
    is_match = torch.allclose(out_ref, out, rtol=0.001, atol=0.001)

    # CHECK: Input 2 - result match: True
    print(f"Input 2 - result match: {is_match}")


def jit_compile_model_function():
    class Model(nn.Module):
        def __init__(self, in_feat, out_feat):
            super().__init__()
            self.net = nn.Linear(in_feat, out_feat)

        def forward(self, x):
            return self.net(x)

    # Compute exected result through the original model.
    batch = 2
    in_feat = 16
    out_feat = 8
    x = torch.randn(batch, in_feat)
    model = Model(in_feat, out_feat)

    # Reference output - invokes PyTorch implementation.
    out_ref = model(x)

    # Reconfigure the model to be compiled using torch.compile.
    model.compile(dynamic=False, backend=cpu_backend(lower_to_llvm))

    # JIT the model using the custom backend.
    # Next invocation goes through MLIR.
    out = model(x)
    is_match = torch.allclose(out_ref, out, rtol=0.01, atol=0.01)

    # CHECK: Compile function - result match: True
    print(f"Compile function - result match: {is_match}")


if __name__ == "__main__":
    # Validate decorator-style API on PyTorch model class.
    jit_compile_model_decorator()
    # Validate function-style API on PyTorch model object.
    jit_compile_model_function()

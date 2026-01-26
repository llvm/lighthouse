# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: torch

import torch
import torch.nn as nn
from torch_mlir.fx import OutputType

from mlir import ir
from mlir.passmanager import PassManager

import lighthouse.runtime as lh_runtime
import lighthouse.utils as lh_utils


def lower_to_llvm(module: ir.Module) -> ir.Module:
    """
    Lower MLIR ops within the module to MLIR LLVM IR dialect.

    A PyTorch models are expected to be imported as Linalg ops using tensors.
    This compilation function preprocesses function signature and applies
    simple lowering to prepare for further jitting.

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
    # Store results in function arguments.
    # The outputs are appended to the function argument list.
    #
    # This transformation is applied to avoid memory management
    # across Python-MLIR boundary.
    pm.add(
        "buffer-results-to-out-params{modify-public-functions=1 hoist-static-allocs=1}"
    )
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


def jit_decorator_model_class():
    # JIT function is used as a class decorator with the MLIR compilation
    # function and target entry MLIR dialect option bound to it.
    #
    # The PyTorch model first must be initialized through its constructor.
    # A standard model object is returned. The decorator injects compilation
    # hooks to redirect execution through MLIR backend.
    #
    # A model object invocation triggers JIT-compilation and execution
    # through MLIR instead of direct PyTorch path.
    @lh_runtime.torch.jit(lower_to_llvm, dialect=OutputType.LINALG_ON_TENSORS)
    class Model(nn.Module):
        def __init__(self, in_features, out_features=3):
            super().__init__()
            self.net = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.net(x)

    # Construct a model object as usual.
    # The PyTorch model is configured to invoke MLIR compilation by default.
    #
    # The positional and keyword arguments are passes to the model's
    # constructor.
    input_size = 16
    output_size = 8
    jit_model: nn.Module = Model(input_size, out_features=output_size)

    # Calling the model's compile method is unnecessary. The JIT decorator
    # configures the model to trigger MLIR compilation by default.
    # However, the compile method can be called safely.
    jit_model.compile()

    # The original PyTorch implementation can still be invoked for verification.
    torch_model = jit_model._call_impl

    # Compute reference result through the original model implementation.
    # This call executes directly through PyTorch.
    batch_size = 2
    input = torch.randn(batch_size, input_size)
    out_ref = torch_model(input)

    # The model has been initialized correctly with the provided sizes.
    # Thus, the result has the size of: [batch_size, output_size]
    #
    # CHECK: Reference out shape: torch.Size([2, 8])
    print(f"Reference out shape: {out_ref.size()}")

    # Call jitted function directly with PyTorch tensors.
    #
    # When only torch.tensor arguments are passed, they are implicitly
    # converted to packed C-type argument before invoking jitted MLIR
    # function.
    #
    # Output parameters are appened to the function arguments - see
    # `lower_to_llvm` compilation function.
    # Thus, an extra empty tensor is passed to store the model result.
    #
    # The PyTorch model is converted to an MLIR module with static shapes.
    # To achieve that, a sample PyTorch input must be provided to allow
    # for shape inference. The importer requires PyTorch-compatible inputs
    # that match original model's invocation singnature.
    # However, the applied MLIR compilation function modifies the resulting
    # jitted function signature.
    # Thus, reference PyTorch inputs are provided through the extra
    # `model_args` argument.
    # See 'lighthouse.runtime.torch.JITModel' for further details.
    out = torch.empty_like(out_ref)
    jit_model(input, out, model_args=(input,))
    is_match = torch.allclose(out_ref, out, rtol=0.01, atol=0.01)

    # CHECK: Tensor args - result match: True
    print(f"Tensor args - result match: {is_match}")

    # Change input shape and recompute reference result.
    input = torch.randn(4, input_size)
    out_ref = torch_model(input)

    # Call jitted function using manually packed C-type arguments.
    #
    # If there are any non-torch.tensor arguments, all inputs get passed
    # directly to the jitted MLIR function.
    # This allows greater control over the exact invocation as the MLIR
    # function signature highly depends on the compilation function.
    out = torch.empty_like(out_ref)
    packed_args = lh_utils.torch.to_packed_args([input, out])
    jit_model(packed_args, model_args=(input,))
    is_match = torch.allclose(out_ref, out, rtol=0.01, atol=0.01)

    # CHECK: Packed args - result match: True
    print(f"Packed args - result match: {is_match}")


def jit_func_model_object():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.matmul(a, b)

    # Construct a model object.
    torch_model = Model()

    # Compute exected result through the original model.
    m_dim = 4
    n_dim = 2
    k_dim = 8
    a = torch.randn(m_dim, k_dim)
    b = torch.randn(k_dim, n_dim)
    out_ref = torch_model(a, b)

    # JIT-compile the created PyTorch model object using
    # MLIR function-style JIT API.
    jit_model: nn.Module = lh_runtime.torch.jit(lower_to_llvm, torch_model)

    # Call directly with PyTorch tensors.
    out = torch.empty_like(out_ref)
    jit_model(a, b, out, model_args=(a, b))
    is_match = torch.allclose(out_ref, out, rtol=0.01, atol=0.01)

    # CHECK: JIT model object - result match: True
    print(f"JIT model object - result match: {is_match}")


if __name__ == "__main__":
    # Validate decorator-style API on PyTorch model class.
    jit_decorator_model_class()
    # Validate function-style API on PyTorch model object.
    jit_func_model_object()

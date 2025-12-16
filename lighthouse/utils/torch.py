import ctypes

import torch
from mlir import ir
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from . import memref as memref_utils


def to_memref(input: torch.Tensor) -> ctypes.Structure:
    """
    Convert a PyTorch tensor into a memref descriptor.

    Args:
        input: PyTorch tensor.
    """
    return get_ranked_memref_descriptor(input.numpy())


def to_packed_args(inputs: list[torch.Tensor]) -> ctypes.Array[ctypes.c_void_p]:
    """
    Convert a list of PyTorch tensors into packed ctype arguments.

    Args:
        inputs: A list of PyTorch tensors.
    """
    memrefs = [to_memref(input) for input in inputs]
    return memref_utils.to_packed_args(memrefs)


def dtype_from_mlir_type(mlir_type: ir.Type):
    """
    Convert an MLIR type to a PyTorch dtype.
    Args:
        mlir_type: An MLIR type (e.g., ir.F32Type, ir.F64Type)
    Returns:
        Corresponding PyTorch dtype
    """
    import torch

    if isinstance(mlir_type, ir.F32Type):
        return torch.float32
    elif isinstance(mlir_type, ir.F64Type):
        return torch.float64
    elif isinstance(mlir_type, ir.F16Type):
        return torch.float16
    elif isinstance(mlir_type, ir.BF16Type):
        return torch.bfloat16
    elif isinstance(mlir_type, ir.IntegerType):
        width = mlir_type.width
        if width == 64:
            return torch.int64
        elif width == 32:
            return torch.int32
        elif width == 16:
            return torch.int16
        elif width == 8:
            return torch.int8
        elif width == 1:
            return torch.bool

    raise ValueError(f"Unsupported MLIR type: {mlir_type}")

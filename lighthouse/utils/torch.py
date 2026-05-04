import ctypes

import torch
from mlir import ir
from mlir.runtime.np_to_memref import (
    BF16,
    F16,
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

from . import memref as memref_utils


def torch_dtype_to_ctype(dtype: torch.dtype):
    """
    Convert a PyTorch dtype to a corresponding ctype.

    Args:
        dtype: PyTorch dtype
    Returns:
        Corresponding ctype
    """
    _dtype_map = {
        torch.float16: F16,
        torch.bfloat16: BF16,
        torch.float32: ctypes.c_float,
        torch.float64: ctypes.c_double,
        torch.int8: ctypes.c_int8,
        torch.int16: ctypes.c_int16,
        torch.int32: ctypes.c_int32,
        torch.int64: ctypes.c_int64,
        torch.uint8: ctypes.c_uint8,
        torch.bool: ctypes.c_bool,
    }
    if dtype not in _dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _dtype_map[dtype]


def to_memref(input: torch.Tensor) -> ctypes.Structure:
    """
    Convert a PyTorch tensor into a memref descriptor.

    Args:
        input: PyTorch tensor.
    """
    ctp = torch_dtype_to_ctype(input.dtype)
    ndim = input.dim()
    data_ptr = input.data_ptr()

    if ndim == 0:
        x = make_zero_d_memref_descriptor(ctp)()
        x.allocated = data_ptr
        x.aligned = ctypes.cast(data_ptr, ctypes.POINTER(ctp))
        x.offset = ctypes.c_longlong(0)
        return x

    x = make_nd_memref_descriptor(ndim, ctp)()
    x.allocated = data_ptr
    x.aligned = ctypes.cast(data_ptr, ctypes.POINTER(ctp))
    x.offset = ctypes.c_longlong(0)
    x.shape = (ctypes.c_longlong * ndim)(*input.shape)
    # PyTorch strides are in element units (unlike NumPy which uses bytes)
    x.strides = (ctypes.c_longlong * ndim)(*input.stride())
    return x


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

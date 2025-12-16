import ctypes
try:
    import torch
except ImportError:
    torch = None

from mlir.runtime.np_to_memref import (
    get_ranked_memref_descriptor,
)
from mlir import ir


def get_packed_arg(ctypes_args) -> list[ctypes.c_void_p]:
    """
    Return a list of packed ctype arguments compatible with
    jitted MLIR function's interface.

    Args:
        ctypes_args: A list of ctype pointer arguments.
    """
    packed_args = (ctypes.c_void_p * len(ctypes_args))()
    for argNum in range(len(ctypes_args)):
        packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
    return packed_args


def memref_to_ctype(memref_desc) -> ctypes._Pointer:
    """
    Convert a memref descriptor into a ctype argument.

    Args:
        memref_desc: An MLIR memref descriptor.
    """
    return ctypes.pointer(ctypes.pointer(memref_desc))


def memrefs_to_packed_args(memref_descs) -> list[ctypes.c_void_p]:
    """
    Convert a list of memref descriptors into packed ctype arguments.

    Args:
        memref_descs: A list of memref descriptors.
    """
    ctype_args = [memref_to_ctype(memref) for memref in memref_descs]
    return get_packed_arg(ctype_args)


if torch is not None:
    def torch_to_memref(input: torch.Tensor) -> ctypes.Structure:
        """
        Convert a PyTorch tensor into a memref descriptor.

        Args:
            input: PyTorch tensor.
        """
        return get_ranked_memref_descriptor(input.numpy())


    def torch_to_packed_args(inputs: list[torch.Tensor]) -> list[ctypes.c_void_p]:
        """
        Convert a list of PyTorch tensors into packed ctype arguments.

        Args:
            inputs: A list of PyTorch tensors.
        """
        memrefs = [torch_to_memref(input) for input in inputs]
        return memrefs_to_packed_args(memrefs)


    def mlir_type_to_torch_dtype(mlir_type: ir.Type):
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

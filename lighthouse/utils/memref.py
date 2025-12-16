import ctypes
from typing import Sequence


def to_ctype(memref_desc) -> ctypes._Pointer:
    """
    Convert a memref descriptor into a ctype argument.

    Args:
        memref_desc: An MLIR memref descriptor.
    """
    return ctypes.pointer(ctypes.pointer(memref_desc))


def get_packed_arg(
    ctypes_args: Sequence[ctypes._Pointer],
) -> ctypes.Array[ctypes.c_void_p]:
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


def to_packed_args(memref_descs) -> ctypes.Array[ctypes.c_void_p]:
    """
    Convert a list of memref descriptors into packed ctype arguments.

    Args:
        memref_descs: A list of memref descriptors.
    """
    ctype_args = [to_ctype(memref) for memref in memref_descs]
    return get_packed_arg(ctype_args)

import ctypes
from collections.abc import Sequence


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


def to_packed_args(args) -> ctypes.Array[ctypes.c_void_p]:
    """
    Convert a list of memref descriptors and/or integers into packed ctype arguments.

    The packed arguments keep the descriptors alive.

    Args:
        args: A list of memref descriptors or integers.
    """
    args = tuple(args)
    # The storage of each argument: an integer, or a pointer to a descriptor.
    slots = (ctypes.c_int64 * len(args))(
        *[arg if isinstance(arg, int) else ctypes.addressof(arg) for arg in args]
    )
    base = ctypes.addressof(slots)
    size = ctypes.sizeof(ctypes.c_int64)
    packed_args = (ctypes.c_void_p * len(args))(
        *[base + i * size for i in range(len(args))]
    )
    # The callee only gets raw addresses.
    # Keep the storage of the arguments alive to prevent them from being
    # garbage collected.
    packed_args._keep_alive = (slots, args)
    return packed_args

import ctypes
from contextlib import contextmanager
from typing import Sequence


@contextmanager
def deallocate_memrefs_on_exit(memref_descs, execution_engine, dealloc_name: str):
    """Context manager that deallocates memref descriptors on exit.

    Invokes the *dealloc_name* function for each descriptor through
    *execution_engine* when the ``with`` block exits.

    Example usage::

        with deallocate_memrefs_on_exit(memrefs, engine, "dealloc_2d"):
            # Use memrefs here
            pass
        # memrefs are deallocated here
    """
    try:
        yield
    finally:
        for desc in memref_descs:
            execution_engine.invoke(dealloc_name, to_ctype(desc))


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

    Args:
        args: A list of memref descriptors or integers.
    """
    ctype_args = []
    for arg in args:
        if isinstance(arg, int):
            ctype_args.append(ctypes.pointer(ctypes.c_int64(arg)))
        else:
            ctype_args.append(to_ctype(arg))
    return get_packed_arg(ctype_args)

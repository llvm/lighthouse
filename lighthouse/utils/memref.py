import ctypes
from contextlib import contextmanager
from typing import Sequence

from mlir.runtime.np_to_memref import ranked_memref_to_numpy


@contextmanager
def mlir_numpy_arrays(
    memref_descs, execution_engine, dealloc_name: str, *, convert: bool = True
):
    """Context manager that deallocates memref descriptors on exit.

    When *convert* is ``True`` (the default), yields a list of numpy arrays
    backed by the memref memory.  When *convert* is ``False``, yields
    ``None`` — useful when you only need the deallocation guarantee (e.g. the
    memrefs may not yet be populated).

    In both cases, when the ``with`` block exits the *dealloc_name* function
    is invoked for each descriptor through *execution_engine*.

    Example usage::

        with mlir_numpy_arrays(memrefs, engine, "dealloc_2d") as arrays:
            A, B, C = arrays
            result = A @ B + C
        # memrefs are deallocated here
    """
    arrays = [ranked_memref_to_numpy([m]) for m in memref_descs] if convert else None
    try:
        yield arrays
    finally:
        for desc in memref_descs:
            try:
                execution_engine.invoke(dealloc_name, to_ctype(desc))
            except Exception:
                pass


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

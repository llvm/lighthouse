import ctypes
import ml_dtypes

import numpy as np
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from lighthouse.utils.memref import to_ctype
from mlir import ir


def numpy_to_ctype(arr: np.ndarray) -> ctypes._Pointer:
    """Convert numpy array to memref and ctypes **void pointer."""
    return to_ctype(get_ranked_memref_descriptor(arr))


def numpy_to_mlir_type(dtype: np.dtype, ctx: ir.Context | None = None) -> ir.Type:
    """
    Convert a numpy dtype into an MLIR type.

    Args:
        dtype: numpy dtype
        ctx: MLIR context (default: current context)
    Returns:
        MLIR type
    """
    if ctx is None:
        ctx = ir.Context.current
    with ctx:
        if dtype == np.float64:
            return ir.F64Type.get()
        if dtype == np.float32:
            return ir.F32Type.get()
        if dtype == np.float16:
            return ir.F16Type.get()
        if dtype == ml_dtypes.bfloat16:
            return ir.BF16Type.get()
        if dtype == np.int64:
            return ir.IntegerType.get(64)
        if dtype == np.int32:
            return ir.IntegerType.get(32)
        if dtype == np.int16:
            return ir.IntegerType.get(16)
        if dtype == np.int8:
            return ir.IntegerType.get(8)
        if dtype == np.bool:
            return ir.IntegerType.get(1)

    raise ValueError(f"Unsupported numpy dtype: {dtype}")

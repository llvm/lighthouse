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
        if dtype == np.bool_:
            return ir.IntegerType.get(1)

    raise ValueError(f"Unsupported numpy dtype: {dtype}")


def mlir_to_numpy_dtype(mlir_type: ir.Type) -> np.dtype:
    """
    Convert an MLIR type into a numpy dtype.

    Args:
        mlir_type: MLIR type
    Returns:
        numpy dtype
    """
    if isinstance(mlir_type, ir.F64Type):
        return np.float64
    if isinstance(mlir_type, ir.F32Type):
        return np.float32
    if isinstance(mlir_type, ir.F16Type):
        return np.float16
    if isinstance(mlir_type, ir.BF16Type):
        return ml_dtypes.bfloat16
    if isinstance(mlir_type, ir.IntegerType):
        width = mlir_type.width
        if width == 64:
            return np.int64
        if width == 32:
            return np.int32
        if width == 16:
            return np.int16
        if width == 8:
            return np.int8
        if width == 1:
            return np.bool_

    raise ValueError(f"Unsupported MLIR type: {mlir_type}")

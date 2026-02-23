import ctypes

import numpy as np
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
from lighthouse.utils.memref import to_ctype


def numpy_to_ctype(arr: np.ndarray) -> ctypes._Pointer:
    """Convert numpy array to memref and ctypes **void pointer."""
    return to_ctype(get_ranked_memref_descriptor(arr))

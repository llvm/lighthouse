# RUN: %PYTHON %s | FileCheck %s

"""Tests for packing JIT function arguments."""

import ctypes
import gc
import weakref

import numpy as np
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from lighthouse.utils.memref import to_packed_args

array = np.arange(12, dtype=np.float32).reshape(3, 4)

# The descriptor is referenced by the packed arguments only.
desc = get_ranked_memref_descriptor(array)
desc_ref = weakref.ref(desc)
packed = to_packed_args([desc, 7, -1])
del desc
gc.collect()
# CHECK: descriptor alive: True
print("descriptor alive:", desc_ref() is not None)

# Each packed argument points to the argument's storage: a pointer to the
# descriptor, or the integer itself.
desc_type = type(desc_ref())
desc_address = ctypes.cast(packed[0], ctypes.POINTER(ctypes.c_void_p)).contents.value
desc = desc_type.from_address(desc_address)
# CHECK: descriptor shape: [3, 4] data: [0.0, 1.0, 2.0, 3.0]
print(
    "descriptor shape:",
    list(desc.shape),
    "data:",
    [desc.aligned[i] for i in range(4)],
)
# CHECK: integers: 7 -1
print(
    "integers:",
    ctypes.cast(packed[1], ctypes.POINTER(ctypes.c_int64)).contents.value,
    ctypes.cast(packed[2], ctypes.POINTER(ctypes.c_int64)).contents.value,
)

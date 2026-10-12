# REQUIRES: torch
# RUN: %PYTHON %s | FileCheck %s

"""Tests for converting PyTorch tensors to memref descriptors."""

import ctypes

import torch
from mlir.runtime.np_to_memref import (
    make_nd_memref_descriptor,
    make_zero_d_memref_descriptor,
)

from lighthouse.utils.torch import _memref_descriptor, to_memref, torch_dtype_to_ctype


def reference_memref(tensor: torch.Tensor) -> ctypes.Structure:
    """The descriptor of ``tensor``, set field by field."""
    ctype = torch_dtype_to_ctype(tensor.dtype)
    ndim = tensor.dim()
    if ndim == 0:
        desc = make_zero_d_memref_descriptor(ctype)()
    else:
        desc = make_nd_memref_descriptor(ndim, ctype)()
        desc.shape = (ctypes.c_longlong * ndim)(*tensor.shape)
        desc.strides = (ctypes.c_longlong * ndim)(*tensor.stride())
    desc.allocated = tensor.data_ptr()
    desc.aligned = ctypes.cast(tensor.data_ptr(), ctypes.POINTER(ctype))
    desc.offset = 0
    return desc


# Descriptors do not keep their tensors alive.
matrix = torch.arange(24, dtype=torch.float32).reshape(4, 6)
tensors = {
    "contiguous": matrix,
    "transposed": matrix.t(),
    "sliced": matrix[1:3, 2:5],
    "rank_3": torch.zeros(2, 3, 4, dtype=torch.int32),
    "bf16": torch.zeros(5, 7, dtype=torch.bfloat16),
    "bool": torch.zeros(8, dtype=torch.bool),
    "scalar": torch.tensor(7.0),
}

# CHECK: contiguous: same bytes True
# CHECK: transposed: same bytes True
# CHECK: sliced: same bytes True
# CHECK: rank_3: same bytes True
# CHECK: bf16: same bytes True
# CHECK: bool: same bytes True
# CHECK: scalar: same bytes True
for name, tensor in tensors.items():
    desc, ref = to_memref(tensor), reference_memref(tensor)
    print(f"{name}: same bytes", bytes(desc) == bytes(ref))

sliced = to_memref(tensors["sliced"])
# CHECK: sliced shape: [2, 3] strides: [6, 1] data: [8.0, 9.0, 10.0]
print(
    "sliced shape:",
    list(sliced.shape),
    "strides:",
    list(sliced.strides),
    "data:",
    [sliced.aligned[i] for i in range(3)],
)
# CHECK: scalar data: 7.0
print("scalar data:", to_memref(tensors["scalar"]).aligned[0])

# Descriptor classes are reused per rank and element type: repeated
# conversions create no new classes (nor ctypes pointer types).
misses = _memref_descriptor.cache_info().misses
for _ in range(100):
    for tensor in tensors.values():
        to_memref(tensor)
# CHECK: new descriptor classes: 0
print("new descriptor classes:", _memref_descriptor.cache_info().misses - misses)
# CHECK: same rank and dtype: True
print(
    "same rank and dtype:",
    type(to_memref(matrix)) is type(to_memref(torch.ones(2, 2))),
)

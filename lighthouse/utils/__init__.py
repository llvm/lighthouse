"""A collection of utility tools"""

from .runtime_args import (
    get_packed_arg,
    memref_to_ctype,
    memrefs_to_packed_args,
)

__all__ = [
    "get_packed_arg",
    "memref_to_ctype",
    "memrefs_to_packed_args",
    "mlir_type_to_torch_dtype",
    "torch_to_memref",
    "torch_to_packed_args",
]

try:
    from .runtime_args import (
        mlir_type_to_torch_dtype,
        torch_to_memref,
        torch_to_packed_args,
    )

    __all__.extend(
        [
            "mlir_type_to_torch_dtype",
            "torch_to_memref",
            "torch_to_packed_args",
        ]
    )
except:
    pass

#!/usr/bin/env python3

"""Thin ctypes helpers for the Level Zero MLIR runtime.

The Xe lowering path ultimately calls into the MLIR Level Zero runtime wrapper
using `mgpuLaunchKernel`, so this module mirrors that ABI instead of managing
Level Zero command lists directly.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections.abc import Iterable, Sequence


ze_kernel_handle_t = ctypes.c_void_p
ze_module_handle_t = ctypes.c_void_p
StreamWrapper = ctypes.c_void_p


def _load_library(lib_path: str | None = None) -> ctypes.CDLL:
    candidates = []
    if lib_path:
        candidates.append(lib_path)
    found = ctypes.util.find_library("mlir_levelzero_runtime")
    if found:
        candidates.append(found)
    candidates.extend(["libmlir_levelzero_runtime.so", "mlir_levelzero_runtime.dll"])

    last_error = None
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            last_error = error

    raise OSError("unable to load the MLIR Level Zero runtime library") from last_error


def _configure_prototypes(lib: ctypes.CDLL) -> None:
    lib.mgpuModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mgpuModuleLoad.restype = ze_module_handle_t

    lib.mgpuModuleLoadJIT.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.mgpuModuleLoadJIT.restype = ze_module_handle_t

    lib.mgpuModuleGetFunction.argtypes = [ze_module_handle_t, ctypes.c_char_p]
    lib.mgpuModuleGetFunction.restype = ze_kernel_handle_t

    lib.mgpuModuleUnload.argtypes = [ze_module_handle_t]
    lib.mgpuModuleUnload.restype = None

    lib.mgpuStreamCreate.argtypes = []
    lib.mgpuStreamCreate.restype = StreamWrapper

    lib.mgpuStreamSynchronize.argtypes = [StreamWrapper]
    lib.mgpuStreamSynchronize.restype = None

    lib.mgpuStreamDestroy.argtypes = [StreamWrapper]
    lib.mgpuStreamDestroy.restype = None

    lib.mgpuLaunchKernel.argtypes = [
        ze_kernel_handle_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        StreamWrapper,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.mgpuLaunchKernel.restype = None


def _as_triplet(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return value, 1, 1

    items = tuple(value)
    if len(items) == 1:
        return int(items[0]), 1, 1
    if len(items) == 2:
        return int(items[0]), int(items[1]), 1
    if len(items) == 3:
        return int(items[0]), int(items[1]), int(items[2])

    raise ValueError(f"{name} must be an int or a sequence of length 1 to 3")


def _argument_pointer(argument):
    if hasattr(argument, "ctypes") and hasattr(argument.ctypes, "data"):
        storage = ctypes.c_void_p(int(argument.ctypes.data))
        return storage, ctypes.byref(storage)

    data_ptr = getattr(argument, "data_ptr", None)
    if callable(data_ptr):
        storage = ctypes.c_void_p(int(data_ptr()))
        return storage, ctypes.byref(storage)

    if isinstance(argument, ctypes._Pointer):  # type: ignore[attr-defined]
        return argument, ctypes.byref(argument)

    if isinstance(
        argument, (ctypes.Array, ctypes.Structure, ctypes.Union, ctypes._SimpleCData)
    ):
        return argument, ctypes.byref(argument)

    raise TypeError(
        "kernel arguments must be ctypes values, ctypes pointers, or objects exposing a NumPy-style ctypes.data"
    )


def _prepare_kernel_arguments(
    arguments: Iterable[object],
) -> tuple[list[object], ctypes.Array[ctypes.c_void_p]]:
    storages: list[object] = []
    pointers: list[int] = []
    for argument in arguments:
        storage, pointer = _argument_pointer(argument)
        storages.append(storage)
        pointers.append(ctypes.cast(pointer, ctypes.c_void_p).value)

    return storages, (ctypes.c_void_p * len(pointers))(*pointers)


def _module_blob_argument(module_blob: object) -> tuple[object, ctypes.c_void_p, int]:
    if isinstance(module_blob, (bytes, bytearray, memoryview)):
        blob_bytes = bytes(module_blob)
        storage = ctypes.create_string_buffer(blob_bytes)
        return storage, ctypes.cast(storage, ctypes.c_void_p), len(blob_bytes)

    if hasattr(module_blob, "ctypes") and hasattr(module_blob.ctypes, "data"):
        blob_size = len(module_blob)
        return module_blob, ctypes.c_void_p(int(module_blob.ctypes.data)), blob_size

    raise TypeError(
        "module_blob must be bytes-like or expose a NumPy-style ctypes.data buffer"
    )


def load_level_zero_module(
    module_blob: object,
    *,
    library_path: str | None = None,
    jit: bool = False,
    opt_level: int = 0,
) -> ze_module_handle_t:
    """Load an embedded MLIR GPU module and return a Level Zero module handle."""

    lib = _load_library(library_path)
    _configure_prototypes(lib)

    storage, blob_ptr, blob_size = _module_blob_argument(module_blob)
    _ = storage
    if jit:
        return lib.mgpuModuleLoadJIT(blob_ptr, opt_level, blob_size)
    return lib.mgpuModuleLoad(blob_ptr, blob_size)


def get_level_zero_kernel_handle(
    module_handle: ze_module_handle_t,
    kernel_name: str,
    *,
    library_path: str | None = None,
) -> ze_kernel_handle_t:
    """Resolve a kernel handle from a loaded module and kernel symbol name."""

    lib = _load_library(library_path)
    _configure_prototypes(lib)
    return lib.mgpuModuleGetFunction(module_handle, kernel_name.encode("utf-8"))


def launch_level_zero_kernel(
    kernel_handle,
    input_arguments: Sequence[object],
    output_arguments: Sequence[object],
    grid_size: int | Sequence[int],
    block_size: int | Sequence[int],
    *,
    shared_mem_bytes: int = 0,
    stream: StreamWrapper | None = None,
    library_path: str | None = None,
) -> StreamWrapper | None:
    """Launch a Level Zero kernel using the MLIR runtime wrapper ABI.

    `input_arguments` and `output_arguments` are forwarded as an array of
    pointers, matching the lowering used by `gpu.launch_func` for Xe.

    If `stream` is omitted, this function creates one, launches the kernel, and
    synchronizes it before returning.
    """

    lib = _load_library(library_path)
    _configure_prototypes(lib)

    created_stream = False
    if stream is None:
        stream = lib.mgpuStreamCreate()
        created_stream = True

    input_storages, input_ptrs = _prepare_kernel_arguments(input_arguments)
    output_storages, output_ptrs = _prepare_kernel_arguments(output_arguments)
    kernel_arg_storages = [*input_storages, *output_storages, input_ptrs, output_ptrs]

    arg_values = [*input_ptrs, *output_ptrs]
    kernel_args = None
    if arg_values:
        kernel_args = (ctypes.c_void_p * len(arg_values))(*arg_values)

    grid_x, grid_y, grid_z = _as_triplet(grid_size, "grid_size")
    block_x, block_y, block_z = _as_triplet(block_size, "block_size")

    try:
        lib.mgpuLaunchKernel(
            kernel_handle,
            grid_x,
            grid_y,
            grid_z,
            block_x,
            block_y,
            block_z,
            shared_mem_bytes,
            stream,
            kernel_args,
            None,
            len(arg_values),
        )
        if created_stream:
            lib.mgpuStreamSynchronize(stream)
        _ = kernel_arg_storages
        return None if created_stream else stream
    finally:
        if created_stream:
            lib.mgpuStreamDestroy(stream)


def load_level_zero_runtime(lib_path: str | None = None) -> ctypes.CDLL:
    return _load_library(lib_path)


def launch_level_zero_module_kernel(
    module_blob: object,
    kernel_name: str,
    input_arguments: Sequence[object],
    output_arguments: Sequence[object],
    grid_size: int | Sequence[int],
    block_size: int | Sequence[int],
    *,
    shared_mem_bytes: int = 0,
    stream: StreamWrapper | None = None,
    library_path: str | None = None,
    jit: bool = False,
    opt_level: int = 0,
) -> StreamWrapper | None:
    """Load a lowered MLIR module, resolve a kernel, and launch it."""

    lib = _load_library(library_path)
    _configure_prototypes(lib)

    storage, blob_ptr, blob_size = _module_blob_argument(module_blob)
    _ = storage
    if jit:
        module_handle = lib.mgpuModuleLoadJIT(blob_ptr, opt_level, blob_size)
    else:
        module_handle = lib.mgpuModuleLoad(blob_ptr, blob_size)

    try:
        kernel_handle = lib.mgpuModuleGetFunction(
            module_handle, kernel_name.encode("utf-8")
        )
        return launch_level_zero_kernel(
            kernel_handle,
            input_arguments,
            output_arguments,
            grid_size,
            block_size,
            shared_mem_bytes=shared_mem_bytes,
            stream=stream,
            library_path=library_path,
        )
    finally:
        lib.mgpuModuleUnload(module_handle)


__all__ = [
    "get_level_zero_kernel_handle",
    "launch_level_zero_kernel",
    "launch_level_zero_module_kernel",
    "load_level_zero_module",
    "load_level_zero_runtime",
]

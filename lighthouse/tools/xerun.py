"""xerun - launch a Xe kernel blob (importable implementation).

Loads a serialized GPU kernel binary blob (such as the one produced by `xeas`)
into the Level Zero runtime and launches its kernel once. Compilation
(IR -> blob) and execution are fully decoupled, so the same blob can be run
repeatedly without recompiling.

The blob is the device kernel embedded by the Xe pipeline in a `gpu.binary` op.
It is loaded with `mgpuModuleLoad` and launched with `mgpuLaunchKernel` through
the helpers in `lighthouse.tools.level_zero_ctypes`. Nothing is recovered from
the blob: the kernel name and the launch grid/block are provided by the caller.

This module is the importable Python API; the command line tool lives in the
``tools/xerun`` executable. The entry point is :func:`xerun`, which loads a blob
and launches its kernel once against the given host buffers:

    from lighthouse.tools.xerun import xerun

    xerun(blob, "payload_kernel", [c, a, b], grid=(8, 4, 1), block=(256, 1, 1))
"""

from lighthouse.tools.level_zero_ctypes import launch_level_zero_module_kernel


def xerun(
    blob,
    kernel_name: str,
    host_buffers: list,
    grid,
    block,
    *,
    library_path: str | None = None,
):
    """Load a GPU kernel blob and launch its kernel once.

    The host buffers are forwarded to the kernel in the given order, so they
    must match the order the kernel expects them. On a shared/unified memory
    setup the kernel reads and writes the buffers in place, so results are
    observed directly in the passed buffers.

    Args:
        blob: The serialized GPU kernel binary (e.g. the output of ``xeas``).
        kernel_name: Name of the kernel symbol inside the blob to launch.
        host_buffers: Buffers passed to the kernel, in the order it expects
            them (numpy arrays or torch tensors).
        grid: Launch grid (number of blocks) as an int or a sequence of up to
            three ints (x, y, z).
        block: Thread block size as an int or a sequence of up to three ints
            (x, y, z).
        library_path: Optional explicit path to the Level Zero runtime library.
    """
    launch_level_zero_module_kernel(
        blob,
        kernel_name,
        host_buffers,
        [],
        grid,
        block,
        library_path=library_path,
    )

"""Launch a XeGPU kernel blob through the Level Zero runtime.

The blob is the device kernel embedded by the Xe pipeline in a `gpu.binary` op.
It is loaded with `mgpuModuleLoad` and launched with `mgpuLaunchKernel` through
the helpers in `lighthouse.execution.xegpu.level_zero_ctypes`. The caller
provides the kernel name and launch grid/block.

Use :func:`xelaunch` to launch synchronously against host buffers:

    from lighthouse.execution.xegpu.xelaunch import xelaunch

    xelaunch(blob, "payload_kernel", [c, a, b], grid=(8, 4, 1), block=(256, 1, 1))
"""

from lighthouse.execution.xegpu.level_zero_ctypes import (
    launch_level_zero_module_kernel,
)


def xelaunch(
    blob,
    kernel_name: str,
    host_buffers: list,
    grid,
    block,
    *,
    library_path: str | None = None,
):
    """Synchronously launch a GPU kernel blob with input buffers only.

    The host buffers are forwarded to the kernel in the given order, so they
    must match the order the kernel expects them. On a shared/unified memory
    setup the kernel reads and writes the buffers in place, so results are
    observed directly in the passed buffers. For output arguments, streams, or
    JIT loading, use :func:`launch_level_zero_module_kernel` directly.

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

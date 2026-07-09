"""xerun - run a prebuilt Xe shared library (importable implementation).

Loads a shared library (.so) such as the one produced by `xeas` and runs its
entry function once. This is the counterpart of `xeas`: compilation (IR -> .so)
and execution are fully decoupled, so the same library can be run repeatedly
without recompiling.

The library must export the MLIR C-interface symbols it is called through:
`_mlir_ciface_<entry-point>` for the entry function and the
`gpu_alloc`/`gpu_copy`/`gpu_dealloc` helpers used to stage buffers on the
device (the kernel always runs on the GPU). The `xeas` output provides all of
these.

This module is the importable Python API; the command line tool lives in the
``tools/xerun`` executable. The entry point is :func:`xerun`, which loads
a shared library and runs its entry function once against the given host
buffers:

    from lighthouse.tools.xerun import xerun

    xerun("payload.so", "payload", [c, a, b])
"""

import numpy as np

from lighthouse.execution.runner import SharedLibraryRunner


def xerun(
    library: str,
    entry_point: str,
    host_input_buffers: list,
    *,
    mem_manager_cls: type | None = None,
    benchmark: str | None = None,
    nruns: int = 100,
    nwarmup: int = 10,
    flops: int | None = None,
):
    """Load a shared library and run its entry function once.

    The entry function is invoked through its MLIR C interface
    (`_mlir_ciface_<entry-point>`); the host buffers are passed positionally in
    the given order, so they must match the order the entry function expects.

    Args:
        library: Filesystem path to the shared library (.so) to run.
        entry_point: Name of the entry function to run.
        host_input_buffers: Buffers passed to the entry function, in the order
            it expects them. When ``mem_manager_cls`` is None they must be numpy
            arrays or torch tensors.
        mem_manager_cls: Optional memory manager class used to stage buffers on
            the device (e.g. ``GPUMemoryManager``). When None the host buffers
            are used directly.
        benchmark: Optional name of the benchmark wrapper function to run
            instead of the entry point.
        nruns: Number of timed benchmark iterations.
        nwarmup: Number of warmup iterations before timing.
        flops: Optional floating-point operation count used to report GFLOPS.
    """
    # Creating the runner (and, when used, the GPU memory manager) needs an
    # active MLIR context to build the element types for the device buffers.
    runner = SharedLibraryRunner(
        library,
        mem_manager_cls=mem_manager_cls,
        benchmark_function_name=benchmark,
    )
    if benchmark is None:
        runner.execute(
            payload_function_name=entry_point,
            host_input_buffers=host_input_buffers,
        )
    else:
        times = runner.benchmark(
            host_input_buffers=host_input_buffers,
            nruns=nruns,
            nwarmup=nwarmup,
        )
        times *= 1e6
        elapsed = float(np.mean(times))
        print(f"time(us): {elapsed:.2f}")
        if flops is not None:
            gflops = flops / (elapsed * 1e-6) / 1e9
            print(f"GFLOPS: {gflops:.2f}")
        else:
            print("GFLOPS: n/a")
    # ``runner`` is dropped when this function returns, unloading the
    # library (see SharedLibraryEngine.__del__) so its ``kernels_unload``
    # finalizer runs now -- while the GPU runtime is still alive -- rather
    # than at process exit where a torn-down Level Zero driver would fault.

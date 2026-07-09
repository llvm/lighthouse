"""xeaddlauncher - add the host launcher for a GPU kernel (importable implementation).

Takes IR that contains an outlined GPU kernel (a `gpu.func` inside a
`gpu.module`) but no host-side entry point, and generates the host launcher
function the rest of the pipeline expects: a `func.func` that launches the
kernel via `gpu.launch_func`, marked `llvm.emit_c_interface` so it is callable
from an external program (see `xeas`/`xerun`).

The matmul example produces this launcher early (during GPU outlining); tools
that join the pipeline on a bare kernel need to add it back. Run this before
`xeas`.

The launch grid (number of blocks) is computed inside the generated IR: the
work-group tile (`wg_tile`) divides the kernel's output dimensions with a
ceiling division (`ceil(size / wg_tile)`), so the grid adapts to the actual
argument sizes at call time. When `wg_tile` is omitted the kernel's
`known_grid_size` annotation is used instead. The block size (threads per
block) is taken from the kernel's `known_block_size` attribute when present,
otherwise it is the `block` argument (default 512 1 1).

The generated launcher is named after the kernel with a trailing `_kernel`
removed (e.g. `payload_kernel` -> `payload`); use `entry_point` to override.
If a host function with that name already exists it is left in place and only
marked `llvm.emit_c_interface`.

Optionally, a benchmarking wrapper can be added on top of the launcher (see
``benchmark``): a `func.func` that calls the entry point in a timed loop,
reusing the shared `bench_wrapper_schedule`. It takes the launcher arguments
plus a timing memref and the number of timed/warmup iterations, so the compiled
library can be benchmarked directly (e.g. via `xerun`).

This module is the importable Python API; the command line tool lives in the
``tools/xeaddlauncher`` executable. The high-level entry point is
:func:`xeaddlauncher`, which parses MLIR text and returns the transformed
module as text:

    from lighthouse.tools.xeaddlauncher import xeaddlauncher

    ir_with_launcher = xeaddlauncher(kernel_text, wg_tile=(256, 256))

:func:`add_launchers` modifies a parsed :class:`mlir.ir.Module` in place (it
requires an active MLIR context) if finer-grained control is needed.
"""

from mlir import ir
from mlir.dialects import arith, func, gpu, memref

from lighthouse.schedule.bench import bench_wrapper_schedule

# Fallback block size used when the kernel has no `known_block_size` annotation.
DEFAULT_BLOCK = [512, 1, 1]


def find_kernels(module: ir.Module):
    """Return the (gpu.module, gpu.func) pairs for every kernel in the module."""
    kernels = []
    for op in module.body.operations:
        if op.operation.name != "gpu.module":
            continue
        for inner in op.regions[0].blocks[0].operations:
            if inner.operation.name == "gpu.func":
                kernels.append((op, inner))
    return kernels


def host_func_names(module: ir.Module) -> set[str]:
    """Names of the top-level host `func.func` ops already in the module."""
    return {
        ir.StringAttr(op.attributes["sym_name"]).value
        for op in module.body.operations
        if op.operation.name == "func.func"
    }


def launcher_arg_types(gpu_func):
    """Expected launcher signature: the kernel arguments."""
    func_type = ir.FunctionType(ir.TypeAttr(gpu_func.attributes["function_type"]).value)
    return list(func_type.inputs)


def get_host_func(module: ir.Module, name: str):
    """Return the top-level host `func.func` op with the given name, or None."""
    for op in module.body.operations:
        if (
            op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == name
        ):
            return op
    return None


def check_and_annotate_launcher(module: ir.Module, gpu_func, name: str) -> None:
    """Validate an existing host function against the launcher signature.

    When a host function with the launcher name is already present, make sure
    its arguments match the kernel arguments before marking it
    `llvm.emit_c_interface`.
    """
    op = get_host_func(module, name)
    expected = launcher_arg_types(gpu_func)
    func_type = ir.FunctionType(ir.TypeAttr(op.attributes["function_type"]).value)
    actual = list(func_type.inputs)
    if actual != expected:
        krn_name = ir.StringAttr(gpu_func.attributes["sym_name"]).value
        raise ValueError(
            f"existing host function '{name}' arguments "
            f"({', '.join(str(t) for t in actual)}) do not match the launcher "
            f"signature for kernel '{krn_name}' "
            f"({', '.join(str(t) for t in expected)})"
        )
    op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def kernel_entry_name(gpu_func) -> str:
    """Default host launcher name: the kernel name without a `_kernel` suffix."""
    name = ir.StringAttr(gpu_func.attributes["sym_name"]).value
    suffix = "_kernel"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def resolve_block(gpu_func, block):
    """Block (thread) size for the launch.

    Uses the kernel's `known_block_size` annotation when present, otherwise the
    provided `block` (three thread dimensions x, y, z).
    """
    if "known_block_size" in gpu_func.attributes:
        return list(ir.DenseI32ArrayAttr(gpu_func.attributes["known_block_size"]))
    return list(block)


def emit_grid(gpu_func, kernel_operands, wg_tile, index_t):
    """Emit the SSA values for the launch grid (number of blocks).

    When `wg_tile` is given, the grid is computed in the IR from the kernel's
    output memref (its first argument, shape M x N) as `ceil(dim / wg_tile)` per
    work-group tile dimension, with a trailing 1 for z. Otherwise the kernel's
    `known_grid_size` annotation is used as a set of constants.
    """
    if wg_tile is not None:
        output = kernel_operands[0]
        grid = []
        for dim, tile in enumerate(wg_tile):
            size = memref.dim(output, arith.constant(index_t, dim))
            grid.append(arith.ceildivui(size, arith.constant(index_t, tile)))
        grid.append(arith.constant(index_t, 1))
        return grid
    if "known_grid_size" in gpu_func.attributes:
        grid = list(ir.DenseI32ArrayAttr(gpu_func.attributes["known_grid_size"]))
        return [arith.constant(index_t, int(v)) for v in grid]
    krn_name = ir.StringAttr(gpu_func.attributes["sym_name"]).value
    raise ValueError(
        f"kernel '{krn_name}' has no 'known_grid_size' annotation; provide "
        "wg_tile so the grid can be computed from the input sizes"
    )


def add_launcher(
    module: ir.Module,
    gpu_mod,
    gpu_func,
    entry_point: str,
    grid=None,
    block=DEFAULT_BLOCK,
    wg_tile=None,
) -> None:
    """Insert a host launcher `func.func` calling `gpu_func` before `gpu_mod`.

    The launcher takes the kernel arguments and forwards them to
    `gpu.launch_func`. The block size is used directly; the grid can be computed in
    the IR from the output dimensions and `wg_tile` (see `emit_grid`).
    """
    assert grid is None or len(grid) == 3, "grid must have three dimensions"
    assert len(block) == 3, "block must have three dimensions"
    mod_name = ir.StringAttr(gpu_mod.attributes["sym_name"]).value
    krn_name = ir.StringAttr(gpu_func.attributes["sym_name"]).value

    block = resolve_block(gpu_func, block)
    arg_types = launcher_arg_types(gpu_func)

    index_t = ir.IndexType.get()
    with ir.InsertionPoint(gpu_mod):
        launcher = func.FuncOp(entry_point, (arg_types, []))
        launcher.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        block_args = launcher.add_entry_block()
        kernel_operands = list(block_args.arguments)
        with ir.InsertionPoint(block_args):
            block_vals = [arith.constant(index_t, int(v)) for v in block]
            grid_vals = (
                [arith.constant(index_t, int(v)) for v in grid]
                if grid is not None
                else emit_grid(gpu_func, kernel_operands, wg_tile, index_t)
            )
            gpu.LaunchFuncOp(
                kernel=[mod_name, krn_name],
                grid_size=tuple(grid_vals),
                block_size=tuple(block_vals),
                kernel_operands=kernel_operands,
            )
            func.ReturnOp([])


def add_benchmark_wrapper(
    module: ir.Module, entry_point: str, bench_name: str = None
) -> None:
    """Wrap the host entry function in a benchmarking function (in place).

    Applies the shared `bench_wrapper_schedule` to ``module``, matching the
    `func.func` named ``entry_point`` (the launcher) and wrapping it in a
    benchmarking function named ``bench_name`` that calls it in a timed loop.

    Requires an active MLIR context with the transform extensions registered.
    """
    schedule = bench_wrapper_schedule(entry_point, bench_name=bench_name)
    schedule.body.operations[0].apply(module.operation)


def add_launchers(
    module: ir.Module,
    *,
    entry_point: str | None = None,
    grid=None,
    block=DEFAULT_BLOCK,
    wg_tile=None,
    benchmark=False,
) -> None:
    """Add host launchers for every kernel in ``module`` (in place).

    Requires an active MLIR context (the module must have been parsed/built in
    one). Marks the module as a `gpu.container_module`, then for each kernel
    either inserts a new launcher or, when a host function with the launcher
    name already exists, validates and annotates it. When ``benchmark`` is set,
    the entry point is additionally wrapped in a benchmarking function.

    Args:
        module: Module containing one or more outlined `gpu.func` kernels.
        entry_point: Name for the generated launcher. Defaults to the kernel
            name without a trailing `_kernel`. Requires a single kernel when set.
        block: Thread block size (x, y, z). Used only when the kernel has no
            `known_block_size` annotation.
        wg_tile: Work-group tile size (M, N). When given, the launch grid is
            computed in the IR as `ceil(output_size / wg_tile)`; otherwise the
            kernel's `known_grid_size` annotation is used.
        benchmark: When provided, wrap the launcher in this benchmarking function.
            Requires a single kernel so the benchmark target is unambiguous.
    """
    kernels = find_kernels(module)
    if not kernels:
        raise ValueError("no 'gpu.func' kernel found in the input IR")
    if entry_point is not None and len(kernels) > 1:
        raise ValueError(
            f"entry_point requires a single kernel, but the input has {len(kernels)}"
        )
    if benchmark and len(kernels) > 1:
        raise ValueError(
            f"benchmark requires a single kernel, but the input has {len(kernels)}"
        )

    module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
    existing = host_func_names(module)
    entries = []
    for gpu_mod, gpu_func in kernels:
        entry = entry_point or kernel_entry_name(gpu_func)
        entries.append(entry)
        if entry in existing:
            # Launcher already present: validate it and make it callable.
            check_and_annotate_launcher(module, gpu_func, entry)
        else:
            add_launcher(
                module,
                gpu_mod,
                gpu_func,
                entry,
                grid=grid,
                block=block,
                wg_tile=wg_tile,
            )

    if benchmark:
        add_benchmark_wrapper(module, entries[0], bench_name=benchmark)


def xeaddlauncher(
    source: str,
    *,
    entry_point: str | None = None,
    grid=None,
    block=DEFAULT_BLOCK,
    wg_tile=None,
    benchmark=False,
) -> str:
    """Add host launchers to an MLIR module given as text.

    This is the high-level, importable counterpart of the command line tool: it
    parses ``source`` in a fresh MLIR context, applies :func:`add_launchers`,
    verifies the result, and returns the transformed module as text.

    Args:
        source: MLIR text containing one or more outlined `gpu.func` kernels.
        entry_point: Name for the generated launcher (see :func:`add_launchers`).
        block: Thread block size (x, y, z).
        wg_tile: Work-group tile size (M, N).
        benchmark: When not False, also wrap the launcher in a benchmarking function with the given name.

    Returns:
        The transformed module as text.
    """
    with ir.Location.unknown():
        module = ir.Module.parse(source)
        add_launchers(
            module,
            entry_point=entry_point,
            grid=grid,
            block=block,
            wg_tile=wg_tile,
            benchmark=benchmark,
        )
        module.operation.verify()
        return str(module)

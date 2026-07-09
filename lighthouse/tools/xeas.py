"""xeas - Xe assembler (importable implementation).

Compiles an "outlined" MLIR payload (the IR produced by the matmul example's
`--dump-kernel=outlined` stage) down to a linkable shared library (.so) using
the same Xe pipeline as the matmul example. The library contains the embedded
GPU binary together with the host-side entry function, so an external program
can link/load against it and call the entry function.

The entry function is exposed through the LLVM C interface, i.e. an external
caller invokes `_mlir_ciface_<entry-point>`.

This module is the importable Python API for the assembler; the command line
tool lives in the ``tools/xeas`` executable. The high-level entry point is
:func:`xeas`, which lowers a payload and returns the compiled shared library as
bytes:

    from lighthouse.tools.xeas import xeas

    params = {"m": 4096, "n": 4096, "k": 4096}
    so_bytes = xeas(source_text, entry_point="payload", params=params)

The entry point, schedule kind and matmul problem sizes are inferred from the
payload when not supplied, so the minimal call is simply
``xeas(source_text)``; any values passed override the inferred defaults.

:func:`lower_payload` (MLIR text -> LLVM-dialect text) and
:func:`compile_shared_library` (LLVM-dialect text -> .so bytes) are available
if finer-grained control is needed.
"""

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlir import ir
from mlir.dialects import memref

from lighthouse.pipeline.driver import TransformDriver, make_function_callable
from lighthouse.schedule.xegpu import (
    ELEMWISE_SCHEDULE,
    MLP_SCHEDULE,
    XeGPUParameterSelector,
    build_payload_schedule,
    xegpu_to_binary,
)
from lighthouse.utils.mlir import get_mlir_library_path


def _parse_input_shape(spec: str) -> list[tuple[list[int], str]]:
    """Parse DIMSxTYPE descriptors into (dims, type) tuples.

    Format: comma-separated descriptors, each `D1xD2x...xTYPE`.
    """
    parsed = []
    for raw in spec.split(","):
        desc = raw.strip()
        if not desc:
            raise ValueError("empty descriptor in --input-shape")
        parts = [p.strip() for p in desc.split("x") if p.strip()]
        if len(parts) < 2:
            raise ValueError(
                f"invalid descriptor '{desc}': expected DIMSxTYPE (e.g. 1024x1024xf16)"
            )
        try:
            dims = [int(p) for p in parts[:-1]]
        except ValueError as exc:
            raise ValueError(
                f"invalid descriptor '{desc}': dimensions must be integers"
            ) from exc
        if not dims or any(d <= 0 for d in dims):
            raise ValueError(f"invalid descriptor '{desc}': dimensions must be > 0")
        parsed.append((dims, parts[-1]))
    return parsed


def _new_static_memref_type(old_type, dims: list[int], elem_name: str):
    """Build a static memref type from an existing memref and descriptor."""
    try:
        old_memref = ir.MemRefType(old_type)
    except Exception as exc:
        raise ValueError(
            f"--input-shape can only rewrite memref arguments, got {old_type}"
        ) from exc
    if old_memref.rank != len(dims):
        raise ValueError(
            f"rank mismatch: descriptor has rank {len(dims)} but argument has rank {old_memref.rank}"
        )
    expected_elem = ir.Type.parse(elem_name)
    if str(old_memref.element_type) != str(expected_elem):
        raise ValueError(
            f"element type mismatch: descriptor uses {expected_elem} but argument has {old_memref.element_type}"
        )
    return ir.MemRefType.get(
        dims, old_memref.element_type, memory_space=old_memref.memory_space
    )


def _set_function_arg_types(func_op, new_input_types):
    """Replace a function's argument types (and block arguments) in place.

    Every use of each old argument is rewired to a freshly created argument of
    the matching new type. ``new_input_types`` must provide one type per
    existing argument; pass the current type for positions that stay unchanged.
    """
    func_type = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)
    block = func_op.regions[0].blocks[0]
    old_args = list(block.arguments)
    if len(old_args) != len(new_input_types):
        raise ValueError(
            "argument attributes are not supported by xeas static-shape rewrite"
        )

    new_args = [
        block.add_argument(new_ty, ir.Location.unknown()) for new_ty in new_input_types
    ]
    for old_arg, new_arg in zip(old_args, new_args):
        old_arg.replace_all_uses_with(new_arg)
    for i in reversed(range(len(old_args))):
        block.erase_argument(i)

    func_op.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get(new_input_types, func_type.results)
    )


def _rewrite_function_with_static_shapes(
    func_op, shape_desc: list[tuple[list[int], str]]
):
    """Rewrite function argument memrefs to static shapes from --input-shape.

    Every use of each argument is rewired to the new static argument. Feeding
    static memrefs into the body (rather than casting back to the original
    dynamic/strided type) lets convert-vector-to-xegpu build XeGPU block
    descriptors directly from the memref instead of from a raw pointer computed
    via memref.extract_strided_metadata; the latter form breaks when the WG
    annotation hoists descriptors above their pointer computation.
    """
    func_type = ir.FunctionType(ir.TypeAttr(func_op.attributes["function_type"]).value)
    old_input_types = list(func_type.inputs)
    if len(old_input_types) != len(shape_desc):
        raise ValueError(
            f"shape count mismatch for '{ir.StringAttr(func_op.attributes['sym_name']).value}': "
            f"got {len(shape_desc)} descriptors for {len(old_input_types)} arguments"
        )

    new_types = [
        _new_static_memref_type(old_ty, dims, elem_name)
        for old_ty, (dims, elem_name) in zip(old_input_types, shape_desc)
    ]
    _set_function_arg_types(func_op, new_types)


def _walk_func_calls(func_op):
    """Return every ``func.call`` op nested anywhere inside ``func_op``."""
    calls = []

    def visit(op):
        if op.operation.name == "func.call":
            calls.append(op)
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner)

    for region in func_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                visit(inner)
    return calls


def _find_callers(module: ir.Module, callee_name: str):
    """Return ``(caller_func_op, call_op)`` for every call to ``callee_name``."""
    callers = []
    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        for call in _walk_func_calls(op):
            if ir.FlatSymbolRefAttr(call.attributes["callee"]).value == callee_name:
                callers.append((op, call))
    return callers


def _reconcile_call_site(caller_op, call_op, target_types) -> bool:
    """Make a call site's operands match the callee's (static) parameter types.

    For each operand whose type differs from the callee parameter it feeds: if
    the operand is a block argument of ``caller_op``, that argument is scheduled
    to become the static type (propagating the shape to the caller's boundary);
    otherwise a ``memref.cast`` refines the locally produced value in place.
    Returns True when the caller's own boundary was made static (so callers of
    ``caller_op`` must be visited in turn).
    """
    caller_args = list(caller_op.regions[0].blocks[0].arguments)
    caller_name = ir.StringAttr(caller_op.attributes["sym_name"]).value
    boundary_updates = {}
    casts = []
    for i, operand in enumerate(call_op.operands):
        want = target_types[i]
        if str(operand.type) == str(want):
            continue
        arg_index = next(
            (j for j, arg in enumerate(caller_args) if operand == arg), None
        )
        if arg_index is None:
            casts.append((i, want))
            continue
        previous = boundary_updates.get(arg_index)
        if previous is not None and str(previous) != str(want):
            raise ValueError(
                f"cannot propagate static shapes: argument {arg_index} of "
                f"'{caller_name}' is passed with conflicting static shapes"
            )
        boundary_updates[arg_index] = want

    for operand_index, want in casts:
        with ir.InsertionPoint(call_op):
            refined = memref.cast(want, call_op.operands[operand_index])
        call_op.operands[operand_index] = refined

    if not boundary_updates:
        return False

    new_types = list(
        ir.FunctionType(ir.TypeAttr(caller_op.attributes["function_type"]).value).inputs
    )
    for arg_index, want in boundary_updates.items():
        new_types[arg_index] = want
    _set_function_arg_types(caller_op, new_types)
    return True


def _propagate_static_args_to_callers(module: ir.Module, func_op):
    """Propagate a function's static argument types upward to its callers.

    Starting at ``func_op`` (the host launcher, whose memref arguments were just
    made static), every ``func.call`` reaching it anywhere in the module is
    reconciled with the new static parameter types. When a caller forwards one
    of its own arguments, that caller's boundary is made static as well and
    propagation continues transitively to its callers.
    """
    worklist = [func_op]
    processed = set()
    while worklist:
        callee = worklist.pop()
        callee_name = ir.StringAttr(callee.attributes["sym_name"]).value
        if callee_name in processed:
            continue
        processed.add(callee_name)
        target_types = list(
            ir.FunctionType(
                ir.TypeAttr(callee.attributes["function_type"]).value
            ).inputs
        )
        for caller_op, call_op in _find_callers(module, callee_name):
            if _reconcile_call_site(caller_op, call_op, target_types):
                worklist.append(caller_op)


def _mark_transfers_in_bounds(module: ir.Module) -> int:
    """Force vector.transfer_read/write ops to be fully in_bounds.

    convert-vector-to-xegpu only lowers transfer ops to XeGPU block
    loads/stores when every dimension is in-bounds; otherwise they stay as
    vector ops and later stages (e.g. the WG anchor-layout annotation that
    expects an xegpu.load producing each dpas operand) fail. Parsed transfer
    ops carry an explicit ``in_bounds`` array that defaults to all-false (the
    all-false form is elided when printed), so we overwrite it with all-true.
    The Xe block-load path assumes in-bounds accesses (boundary_check = false),
    so this is consistent with the rest of the pipeline. Returns the number of
    ops updated.
    """
    count = 0

    def visit(op):
        nonlocal count
        name = op.operation.name
        if name in ("vector.transfer_read", "vector.transfer_write"):
            vec_type = (
                op.results[0].type
                if name == "vector.transfer_read"
                else op.operands[0].type
            )
            rank = ir.VectorType(vec_type).rank
            all_true = ir.ArrayAttr.get([ir.BoolAttr.get(True)] * rank)
            current = (
                op.attributes["in_bounds"] if "in_bounds" in op.attributes else None
            )
            if current is None or str(current) != str(all_true):
                op.attributes["in_bounds"] = all_true
                count += 1
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    visit(inner)

    for op in module.body.operations:
        visit(op)
    return count


def _get_entry_func(module: ir.Module, entry_point: str):
    for op in module.body.operations:
        if (
            op.operation.name == "func.func"
            and ir.StringAttr(op.attributes["sym_name"]).value == entry_point
        ):
            return op
    return None


def _get_gpu_funcs(module: ir.Module):
    gpu_funcs = []
    for op in module.body.operations:
        if op.operation.name != "gpu.module":
            continue
        for inner in op.regions[0].blocks[0].operations:
            if inner.operation.name == "gpu.func":
                gpu_funcs.append(inner)
    return gpu_funcs


@dataclass
class OutlinedPayloadInfo:
    """Information inferred from an outlined MLIR payload.

    Every field may be ``None`` when it cannot be recovered from the IR; callers
    are expected to supply the missing values explicitly (and may override any
    inferred value).
    """

    entry_point: str | None = None
    schedule_kind: str | None = None
    m: int | None = None
    n: int | None = None
    k: int | None = None
    transpose_a: bool = False
    transpose_b: bool = False

    def default_params(self) -> dict:
        """Return the schedule parameters implied by the inferred payload.

        Only fully determined matmul sizes are returned; an empty dict is
        returned when the sizes could not be inferred.
        """
        if None in (self.m, self.n, self.k):
            return {}
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
        }


def _contains_op(root, op_name: str) -> bool:
    """Return True if ``root`` contains an op named ``op_name`` anywhere."""
    for region in root.operation.regions:
        for block in region.blocks:
            for op in block.operations:
                if op.operation.name == op_name or _contains_op(op, op_name):
                    return True
    return False


def _find_entry_point(module: ir.Module) -> str | None:
    """Infer the host entry function of an outlined payload.

    The entry function is the ``func.func`` that launches the kernel, i.e. the
    one containing a ``gpu.launch_func`` op. When the launcher cannot be
    identified unambiguously, ``None`` is returned.
    """
    candidates = [
        ir.StringAttr(op.attributes["sym_name"]).value
        for op in module.body.operations
        if op.operation.name == "func.func" and _contains_op(op, "gpu.launch_func")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _detect_schedule_kind(module: ir.Module) -> str | None:
    """Infer the schedule kind from the outlined kernel body.

    A ``vector.contract`` in the kernel identifies a matmul/MLP payload;
    otherwise the presence of a ``gpu.func`` indicates an elementwise payload.
    Returns ``None`` when no kernel is found.
    """
    gpu_funcs = _get_gpu_funcs(module)
    if not gpu_funcs:
        return None
    for gpu_func in gpu_funcs:
        if _contains_op(gpu_func, "vector.contract"):
            return MLP_SCHEDULE
    return ELEMWISE_SCHEDULE


def _static_2d_shape(type_) -> list[int] | None:
    """Return the static 2D shape of a memref type, or ``None`` otherwise."""
    try:
        memref_type = ir.MemRefType(type_)
    except (ValueError, TypeError):
        return None
    if memref_type.rank != 2:
        return None
    shape = list(memref_type.shape)
    if any(ir.ShapedType.is_dynamic_size(dim) for dim in shape):
        return None
    return shape


def _infer_matmul_sizes(entry_func) -> tuple[int, int, int, bool, bool] | None:
    """Infer ``(m, n, k, transpose_a, transpose_b)`` from the entry signature.

    The outlined matmul launcher follows the ``payload(C, A, B)`` convention,
    where ``C`` is ``M x N`` and ``A``/``B`` are the (possibly transposed)
    operands. ``M`` and ``N`` come from ``C``; ``K`` and the transpose flags are
    recovered by matching ``A``/``B`` against the derived ``M``/``N``. Returns
    ``None`` when the signature does not match a static 2D matmul.
    """
    func_type = ir.FunctionType(
        ir.TypeAttr(entry_func.attributes["function_type"]).value
    )
    inputs = list(func_type.inputs)
    if len(inputs) < 3:
        return None
    c_shape = _static_2d_shape(inputs[0])
    a_shape = _static_2d_shape(inputs[1])
    b_shape = _static_2d_shape(inputs[2])
    if not (c_shape and a_shape and b_shape):
        return None

    m, n = c_shape
    # A is (M, K) when not transposed, (K, M) when transposed.
    if a_shape[0] == m:
        k, transpose_a = a_shape[1], False
    elif a_shape[1] == m:
        k, transpose_a = a_shape[0], True
    else:
        return None
    # Validate B against the derived K and N.
    if b_shape == [k, n]:
        transpose_b = False
    elif b_shape == [n, k]:
        transpose_b = True
    else:
        return None
    return m, n, k, transpose_a, transpose_b


def inspect_outlined_payload(module: ir.Module) -> OutlinedPayloadInfo:
    """Inspect an outlined MLIR payload and infer its schedule metadata.

    Recovers the entry point, schedule kind and (for matmul payloads) the
    problem sizes and transpose flags directly from the IR, so callers do not
    need to restate information already encoded in the payload.
    """
    entry_point = _find_entry_point(module)
    schedule_kind = _detect_schedule_kind(module)
    info = OutlinedPayloadInfo(entry_point=entry_point, schedule_kind=schedule_kind)
    if entry_point is not None and schedule_kind == MLP_SCHEDULE:
        entry_func = _get_entry_func(module, entry_point)
        sizes = _infer_matmul_sizes(entry_func) if entry_func is not None else None
        if sizes is not None:
            info.m, info.n, info.k, info.transpose_a, info.transpose_b = sizes
    return info


# Tile parameters consumed by the XeGPU WG annotation stage. When only some of
# these are supplied, the schedule's own selector would overwrite them all, so
# we pre-fill the full set from the selector and re-apply the overrides.
_TILE_PARAM_NAMES = (
    "wg_m",
    "wg_n",
    "sg_m",
    "sg_n",
    "k_tile",
    "load_a_m",
    "load_a_k",
    "load_b_k",
    "load_b_n",
    "prefetch_a_m",
    "prefetch_a_k",
    "prefetch_b_k",
    "prefetch_b_n",
    "prefetch_a_nb",
    "prefetch_b_nb",
)


def _resolve_mlp_tile_params(params: dict) -> dict:
    """Complete a partial set of matmul tile parameters.

    If the caller provided some (but not all) tile parameters, fill the missing
    ones from the parameter selector and re-apply the caller's overrides, so a
    partial override is not discarded by the schedule's own selection step.
    Fully specified or empty tile-parameter sets are returned unchanged.
    """
    overrides = {name: params[name] for name in _TILE_PARAM_NAMES if name in params}
    if not overrides or all(name in params for name in _TILE_PARAM_NAMES):
        return params
    selector = XeGPUParameterSelector(device=params.get("device"))
    filled = selector.get_parameters(
        (params["m"], params["n"], params["k"]),
        params.get("transpose_a", False),
        params.get("transpose_b", False),
    )
    return {**params, **filled, **overrides}


def _llvm_tool(name):
    """Return the path to an LLVM tool from the install that provides the MLIR
    Python bindings (``<install>/bin``)."""
    tool = Path(get_mlir_library_path()).parent / "bin" / name
    if not tool.is_file():
        raise RuntimeError(f"could not find '{name}' in {tool.parent}")
    return str(tool)


# GPU runtime libraries the generated kernel calls into (mgpuLaunchKernel,
# rtclock, the gpu alloc/copy helpers, ...). They are recorded as dependencies
# of the .so so it loads standalone instead of relying on its loader to already
# provide these symbols.
RUNTIME_LIBS = ["libmlir_levelzero_runtime.so", "libmlir_c_runner_utils.so"]


def _runtime_libs():
    """Resolve the runtime dependency libraries to absolute paths in the MLIR
    library directory."""
    lib_dir = Path(get_mlir_library_path())
    resolved = []
    for name in RUNTIME_LIBS:
        path = lib_dir / name
        if not path.is_file():
            raise RuntimeError(f"could not find runtime library {path}")
        resolved.append(str(path))
    return resolved


def lower_payload(
    source: str,
    entry_point: str | None = None,
    params: dict | None = None,
    *,
    input_shape: str | None = None,
    assume_in_bounds: bool = True,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
) -> str:
    """Lower an outlined MLIR payload to an LLVM-dialect module (as text).

    Resumes the Xe pipeline from the 'outlined' stage and lowers the kernel all
    the way to an embedded GPU binary inside an LLVM-dialect module. This is the
    context-bound part of :func:`xeas`; see it for a description of the
    ``input_shape`` and ``assume_in_bounds`` behaviour.

    The entry point, schedule kind and matmul problem sizes are inferred from
    the payload when not supplied; any value passed in ``entry_point`` or
    ``params`` overrides the inferred default.

    Args:
        source: MLIR text at the 'outlined' stage.
        entry_point: Name of the entry function to expose (callable as
            ``_mlir_ciface_<entry-point>``). Inferred from the payload's kernel
            launcher when ``None``.
        params: Schedule parameter dict. Missing matmul sizes are inferred from
            the payload; provided values override the inferred ones.
        input_shape: Optional comma-separated ``DIMSxTYPE`` descriptors used to
            rewrite launcher/kernel argument memrefs to static shapes.
        assume_in_bounds: Mark transfer ops omitting ``in_bounds`` as fully
            in-bounds so they lower to XeGPU block loads/stores.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.

    Returns:
        The lowered LLVM-dialect module as text.
    """
    with ir.Location.unknown():
        module = ir.Module.parse(source)

        # Recover entry point, schedule kind and matmul sizes from the payload,
        # then let caller-provided values override the inferred defaults.
        info = inspect_outlined_payload(module)
        if entry_point is None:
            entry_point = info.entry_point
        if entry_point is None:
            raise ValueError(
                "could not infer the entry point from the payload; pass "
                "entry_point explicitly"
            )
        schedule_kind = info.schedule_kind
        if schedule_kind is None:
            raise ValueError(
                "could not infer the schedule kind from the payload; the "
                "module does not contain a gpu kernel"
            )
        params = {**info.default_params(), **(params or {})}
        if schedule_kind == MLP_SCHEDULE:
            if not all(params.get(dim) for dim in ("m", "n", "k")):
                raise ValueError(
                    "could not infer the matmul sizes (m, n, k) from the "
                    "payload; pass them explicitly"
                )
            params = _resolve_mlp_tile_params(params)

        if input_shape:
            try:
                shapes = _parse_input_shape(input_shape)
                entry_func = _get_entry_func(module, entry_point)
                if entry_func is None:
                    raise ValueError(
                        f"entry function '{entry_point}' was not found for "
                        "--input-shape rewriting"
                    )
                _rewrite_function_with_static_shapes(entry_func, shapes)
                for gpu_func in _get_gpu_funcs(module):
                    _rewrite_function_with_static_shapes(gpu_func, shapes)
                # Start at the host launcher and push the static argument types
                # transitively up to every caller in the module.
                _propagate_static_args_to_callers(module, entry_func)
            except Exception as exc:
                raise ValueError(f"invalid --input-shape rewrite: {exc}") from exc

        if assume_in_bounds:
            marked = _mark_transfers_in_bounds(module)
            if marked:
                print(
                    f"xeas: marked {marked} vector transfer op(s) as in_bounds",
                    file=sys.stderr,
                )

        # Expose the entry function before LLVM lowering so a C-interface
        # wrapper (_mlir_ciface_<entry-point>) is emitted and the symbol is
        # linkable from an external program.
        make_function_callable(module, entry_point)

        schedules = [
            build_payload_schedule(
                schedule_kind,
                [params],
                payload_func_name=entry_point,
                start_at_stage="outlined",
            ),
            xegpu_to_binary(
                xegpu_op_level=xegpu_op_level,
                large_register_file=large_register_file,
            ),
        ]
        lowered = TransformDriver(schedules).apply(module)
        return str(lowered)


def compile_shared_library(llvm_dialect_text: str, opt_level: int = 3) -> bytes:
    """Turn an LLVM-dialect MLIR module into shared library bytes (.so).

    The LLVM steps use binaries from the LLVM install that provides the MLIR
    Python bindings:
        mlir-translate : LLVM-dialect MLIR -> LLVM IR
        llc            : LLVM IR          -> relocatable object
    The object is then linked into a shared library with the system C compiler
    (cc); this LLVM install ships no linker of its own. The GPU runtime
    libraries are linked in (and rpath'd) so the kernel's runtime symbols
    (mgpu*, rtclock, ...) resolve when the library is loaded.

    A static toolchain is used on purpose (rather than the MLIR JIT) so the GPU
    runtime is never loaded into this process: the JIT would load and later
    tear down the device module, which crashes on hosts without a usable GPU.

    Returns:
        The compiled shared library as bytes.
    """
    mlir_translate = _llvm_tool("mlir-translate")
    llc = _llvm_tool("llc")
    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        raise RuntimeError("could not find 'cc' or 'gcc' on $PATH")
    runtime_libs = _runtime_libs()

    llvm_ir = subprocess.run(
        [mlir_translate, "--mlir-to-llvmir"],
        input=llvm_dialect_text.encode(),
        stdout=subprocess.PIPE,
        check=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        obj = Path(tmp) / "xeas.o"
        subprocess.run(
            [
                llc,
                f"-O{opt_level}",
                "-filetype=obj",
                "-relocation-model=pic",
                "-o",
                str(obj),
                "-",
            ],
            input=llvm_ir.stdout,
            check=True,
        )

        # The linker seeks within its output file, so it cannot write directly
        # to a pipe: always link into a seekable temporary file and return its
        # bytes.
        out_path = Path(tmp) / "xeas.so"
        lib_dir = str(Path(get_mlir_library_path()))
        # '--as-needed' records a NEEDED dependency only for the runtime
        # libraries the kernel actually references, not all of them.
        subprocess.run(
            [cc, "-shared", "-o", str(out_path), str(obj), "-Wl,--as-needed"]
            + runtime_libs
            + [f"-Wl,-rpath,{lib_dir}"],
            check=True,
        )
        return out_path.read_bytes()


def xeas(
    source: str,
    entry_point: str | None = None,
    params: dict | None = None,
    *,
    input_shape: str | None = None,
    assume_in_bounds: bool = True,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
    opt_level: int = 3,
) -> bytes:
    """Compile an outlined MLIR payload into a shared library (.so).

    This is the high-level, importable counterpart of the command line tool: it
    lowers ``source`` with :func:`lower_payload` and compiles the result with
    :func:`compile_shared_library`, returning the library bytes.

    The entry point, schedule kind and matmul problem sizes are inferred from
    the payload when not supplied; any value passed in ``entry_point`` or
    ``params`` overrides the inferred default.

    Args:
        source: MLIR text at the 'outlined' stage.
        entry_point: Name of the entry function to expose (callable as
            ``_mlir_ciface_<entry-point>``). Inferred from the payload's kernel
            launcher when ``None``.
        params: Schedule parameter dict. Programmatic callers can pass a dict
            such as ``{"m": M, "n": N, "k": K}`` plus any tile overrides; any
            values not supplied are inferred from the payload.
        input_shape: Optional comma-separated ``DIMSxTYPE`` descriptors used to
            rewrite launcher/kernel argument memrefs to static shapes.
        assume_in_bounds: Mark transfer ops omitting ``in_bounds`` as fully
            in-bounds so they lower to XeGPU block loads/stores.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.
        opt_level: Optimization level for the generated code.

    Returns:
        The compiled shared library as bytes.
    """
    llvm_dialect_text = lower_payload(
        source,
        entry_point,
        params,
        input_shape=input_shape,
        assume_in_bounds=assume_in_bounds,
        xegpu_op_level=xegpu_op_level,
        large_register_file=large_register_file,
    )
    return compile_shared_library(llvm_dialect_text, opt_level)

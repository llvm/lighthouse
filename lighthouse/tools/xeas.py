"""xeas - Xe assembler (importable implementation).

Compiles an "outlined" MLIR payload down to a serialized GPU kernel binary blob.

The input is an outlined kernel: a ``gpu.module`` containing a single
``gpu.func`` (the vectorized kernel body), and nothing else -- no host
``func.func`` and no ``gpu.launch_func`` launcher. This is the IR produced by
the XeGPU pipeline's ``outlined`` stage.

xeas resumes the XeGPU pipeline from that stage and lowers the kernel to a
``gpu.binary`` op, then returns the embedded device object -- the exact byte
string the Level Zero runtime loads at run time (``mgpuModuleLoad``). ``xerun``
launches the kernel directly from this blob, so no host-side launcher or shared
library is produced.

Lowering parameters that cannot be recovered from the IR are passed in via
``params`` (e.g. the matmul sizes ``m``/``n``/``k`` and any XeGPU tile
overrides); matmul sizes are inferred from the kernel signature when omitted.

This module is the importable Python API; the command line tool lives in the
``tools/xeas`` executable. The high-level entry point is :func:`xeas`:

    from lighthouse.tools.xeas import xeas

    blob = xeas(outlined_mlir, params={"m": 2048, "n": 4096, "k": 8192})
"""

import sys

from mlir import ir
from mlir.dialects import func, gpu

from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import (
    ELEMWISE_SCHEDULE,
    MLP_SCHEDULE,
    XeGPUParameterSelector,
    build_payload_schedule,
    xegpu_to_binary,
)

# The XeGPU schedule locates the payload module by matching a top-level
# ``func.func`` by name. The outlined input has none (only a ``gpu.func``), so a
# throwaway function under this name is injected for the schedule to anchor on.
_SCHEDULE_ANCHOR = "__xeas_schedule_anchor"

# Tile parameters consumed by the XeGPU WG annotation stage. When only some of
# these are supplied, the schedule's own selector would overwrite them all, so
# the full set is pre-filled from the selector and the overrides re-applied.
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


def _get_gpu_funcs(module: ir.Module) -> list:
    """Return every ``gpu.func`` op nested in a ``gpu.module`` of the module."""
    gpu_funcs = []
    for op in module.body.operations:
        if op.operation.name != "gpu.module":
            continue
        for inner in op.regions[0].blocks[0].operations:
            if inner.operation.name == "gpu.func":
                gpu_funcs.append(inner)
    return gpu_funcs


def _parse_input_shape(spec: str) -> list[tuple[list[int], str]]:
    """Parse DIMSxTYPE descriptors into (dims, type) tuples.

    Format: comma-separated descriptors, each ``D1xD2x...xTYPE``.
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
    """Rewrite a kernel's argument memrefs to static shapes from --input-shape.

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


def _contains_op(root, op_name: str) -> bool:
    """Return True if ``root`` contains an op named ``op_name`` anywhere."""
    for region in root.operation.regions:
        for block in region.blocks:
            for op in block.operations:
                if op.operation.name == op_name or _contains_op(op, op_name):
                    return True
    return False


def _detect_schedule_kind(gpu_funcs: list) -> str:
    """Infer the schedule kind from the outlined kernel body.

    A ``vector.contract`` in the kernel identifies a matmul/MLP payload;
    otherwise it is treated as an elementwise payload.
    """
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


def _infer_matmul_sizes(gpu_func) -> tuple[int, int, int, bool, bool] | None:
    """Infer ``(m, n, k, transpose_a, transpose_b)`` from the kernel signature.

    The outlined matmul kernel follows the ``kernel(C, A, B)`` convention, where
    ``C`` is ``M x N`` and ``A``/``B`` are the (possibly transposed) operands.
    ``M`` and ``N`` come from ``C``; ``K`` and the transpose flags are recovered
    by matching ``A``/``B`` against the derived ``M``/``N``. Returns ``None``
    when the signature does not match a static 2D matmul.
    """
    func_type = ir.FunctionType(ir.TypeAttr(gpu_func.attributes["function_type"]).value)
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


def _prepare_mlp_params(gpu_funcs: list, params: dict) -> dict:
    """Complete the matmul lowering parameters for the MLP schedule.

    Missing ``m``/``n``/``k`` (and the transpose flags) are inferred from the
    kernel signature. A partial set of tile parameters is completed from the
    parameter selector while preserving the caller's overrides, so a partial
    override is not discarded by the schedule's own selection step.
    """
    params = dict(params)
    if not all(dim in params for dim in ("m", "n", "k")):
        sizes = _infer_matmul_sizes(gpu_funcs[0])
        if sizes is None:
            raise ValueError(
                "could not infer the matmul sizes from the kernel signature; "
                "pass m, n and k in params"
            )
        m, n, k, transpose_a, transpose_b = sizes
        params.setdefault("m", m)
        params.setdefault("n", n)
        params.setdefault("k", k)
        params.setdefault("transpose_a", transpose_a)
        params.setdefault("transpose_b", transpose_b)
    params.setdefault("transpose_a", False)
    params.setdefault("transpose_b", False)

    overrides = {name: params[name] for name in _TILE_PARAM_NAMES if name in params}
    if overrides and not all(name in params for name in _TILE_PARAM_NAMES):
        selector = XeGPUParameterSelector(device=params.get("device"))
        filled = selector.get_parameters(
            (params["m"], params["n"], params["k"]),
            params["transpose_a"],
            params["transpose_b"],
        )
        params = {**params, **filled, **overrides}
    return params


def _mark_transfers_in_bounds(module: ir.Module) -> int:
    """Force ``vector.transfer_read``/``transfer_write`` ops to be in_bounds.

    convert-vector-to-xegpu only lowers transfer ops to XeGPU block
    loads/stores when every dimension is in-bounds; otherwise they stay as
    vector ops and later stages fail. Parsed transfer ops carry an explicit
    ``in_bounds`` array that defaults to all-false (elided when printed), so it
    is overwritten with all-true. The Xe block-load path assumes in-bounds
    accesses, so this is consistent with the rest of the pipeline. Returns the
    number of ops updated.
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


def _inject_schedule_anchor(module: ir.Module) -> str:
    """Insert a throwaway ``func.func`` for the XeGPU schedule to anchor on.

    The outlined input contains only a ``gpu.module``/``gpu.func`` and no host
    ``func.func``, but the XeGPU schedule locates the payload module by matching
    a ``func.func`` by name. This adds an empty function it can grab; the
    function is never called and is lowered to a dead, empty ``llvm.func`` by
    the binary lowering (only the ``gpu.binary`` object is extracted).
    """
    with ir.InsertionPoint(module.body):
        anchor = func.FuncOp(_SCHEDULE_ANCHOR, ([], []))
        with ir.InsertionPoint(anchor.add_entry_block()):
            func.ReturnOp([])
    return _SCHEDULE_ANCHOR


def _select_gpu_object(objects: ir.ArrayAttr, large_register_file: bool):
    """Pick the ``#gpu.object`` matching the large-register-file setting.

    With a single object the choice is unambiguous; otherwise the object whose
    target carries (or omits) the ``-ze-opt-large-register-file`` IGC option is
    selected to match ``large_register_file``, falling back to the first object.
    """
    if len(objects) == 1:
        return gpu.ObjectAttr(objects[0])

    def uses_lrf(obj) -> bool:
        return "large-register-file" in str(gpu.ObjectAttr(obj).target)

    for obj in objects:
        if uses_lrf(obj) == large_register_file:
            return gpu.ObjectAttr(obj)
    return gpu.ObjectAttr(objects[0])


def extract_gpu_binary(module: ir.Module, *, large_register_file: bool = True) -> bytes:
    """Extract the serialized GPU kernel binary from a lowered module.

    The XeGPU-to-binary pipeline embeds the device kernel in a ``gpu.binary`` op
    as one or more ``#gpu.object`` attributes (one per target attached to the
    ``gpu.module``). This returns the raw object bytes -- the blob the Level Zero
    runtime loads at run time via ``mgpuModuleLoad``.

    When several objects are present (e.g. a plain target and one built with the
    large register file IGC option), the object matching ``large_register_file``
    is returned.
    """
    binaries = [
        op for op in module.body.operations if op.operation.name == "gpu.binary"
    ]
    if not binaries:
        raise ValueError("no 'gpu.binary' op found in the lowered module")
    if len(binaries) > 1:
        raise ValueError(f"expected a single 'gpu.binary' op, found {len(binaries)}")
    objects = ir.ArrayAttr(binaries[0].attributes["objects"])
    if len(objects) == 0:
        raise ValueError("'gpu.binary' op has no embedded objects")
    return _select_gpu_object(objects, large_register_file).object


def lower_payload(
    source: str,
    params: dict | None = None,
    *,
    input_shape: str | None = None,
    assume_in_bounds: bool = True,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
) -> ir.Module:
    """Lower an outlined MLIR kernel to a module containing its ``gpu.binary``.

    Resumes the XeGPU pipeline from the ``outlined`` stage (a ``gpu.module`` with
    a single ``gpu.func``) and lowers it to an embedded GPU binary. This is the
    context-bound part of :func:`xeas` and must run inside an active MLIR
    context.

    Args:
        source: MLIR text at the ``outlined`` stage.
        params: Lowering parameters. For a matmul/MLP kernel this holds the
            problem sizes ``m``/``n``/``k`` (inferred from the kernel signature
            when omitted) plus optional transpose flags, ``device`` and XeGPU
            tile overrides.
        input_shape: Optional comma-separated ``DIMSxTYPE`` descriptors, one per
            kernel argument in order. When given, the ``gpu.func`` kernel's
            (possibly dynamically shaped) memref arguments are rewritten to
            these static shapes.
        assume_in_bounds: Mark ``vector.transfer`` ops omitting ``in_bounds`` as
            fully in-bounds so they lower to XeGPU block loads/stores.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.

    Returns:
        The lowered module containing the embedded ``gpu.binary`` kernel.
    """
    params = dict(params or {})
    with ir.Location.unknown():
        module = ir.Module.parse(source)

        gpu_funcs = _get_gpu_funcs(module)
        if not gpu_funcs:
            raise ValueError("input IR contains no 'gpu.func' kernel to lower")

        if input_shape:
            try:
                shapes = _parse_input_shape(input_shape)
                for gpu_func in gpu_funcs:
                    _rewrite_function_with_static_shapes(gpu_func, shapes)
            except Exception as exc:
                raise ValueError(f"invalid --input-shape rewrite: {exc}") from exc

        schedule_kind = _detect_schedule_kind(gpu_funcs)
        if schedule_kind == MLP_SCHEDULE:
            params = _prepare_mlp_params(gpu_funcs, params)

        if assume_in_bounds:
            marked = _mark_transfers_in_bounds(module)
            if marked:
                print(
                    f"xeas: marked {marked} vector transfer op(s) as in_bounds",
                    file=sys.stderr,
                )

        anchor = _inject_schedule_anchor(module)

        schedules = [
            build_payload_schedule(
                schedule_kind,
                [params],
                payload_func_name=anchor,
                start_at_stage="outlined",
            ),
            xegpu_to_binary(
                xegpu_op_level=xegpu_op_level,
                large_register_file=large_register_file,
            ),
        ]
        return TransformDriver(schedules).apply(module)


def xeas(
    source: str,
    params: dict | None = None,
    *,
    input_shape: str | None = None,
    assume_in_bounds: bool = True,
    xegpu_op_level: str = "workgroup",
    large_register_file: bool = True,
) -> bytes:
    """Compile an outlined MLIR kernel into a GPU kernel binary blob.

    Lowers ``source`` with :func:`lower_payload` and extracts the embedded
    device kernel with :func:`extract_gpu_binary`, returning the blob bytes --
    the exact byte string the Level Zero runtime loads at run time
    (``mgpuModuleLoad``). ``xerun`` launches the kernel directly from it.

    Args:
        source: MLIR text at the ``outlined`` stage.
        params: Lowering parameters (see :func:`lower_payload`).
        input_shape: Optional comma-separated ``DIMSxTYPE`` descriptors used to
            rewrite the ``gpu.func`` kernel's memref arguments to static shapes.
        assume_in_bounds: Mark transfer ops omitting ``in_bounds`` as fully
            in-bounds so they lower to XeGPU block loads/stores.
        xegpu_op_level: Initial XeGPU operation level for the lowering pipeline.
        large_register_file: Enable the large register file IGC option.

    Returns:
        The serialized GPU kernel binary as bytes.
    """
    module = lower_payload(
        source,
        params,
        input_shape=input_shape,
        assume_in_bounds=assume_in_bounds,
        xegpu_op_level=xegpu_op_level,
        large_register_file=large_register_file,
    )
    return extract_gpu_binary(module, large_register_file=large_register_file)

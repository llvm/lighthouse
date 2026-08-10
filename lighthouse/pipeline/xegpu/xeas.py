"""xeas - Xe assembler (importable implementation).

Compiles an "outlined" MLIR payload down to a serialized GPU kernel binary blob.

The input is an outlined kernel: a ``gpu.module`` containing a single
``gpu.func`` (the vectorized kernel body), and nothing else -- no host
``func.func`` and no ``gpu.launch_func`` launcher. This is the IR produced by
the XeGPU pipeline's ``outlined`` stage.

xeas resumes the XeGPU pipeline from that stage and lowers the kernel to a
``gpu.binary`` op, then returns the embedded device object -- the exact byte
string the Level Zero runtime loads at run time (``mgpuModuleLoad``). The blob
can be passed to :func:`lighthouse.execution.xegpu.xelaunch.xelaunch` without
producing a host-side launcher or shared library.

The high-level entry point is :func:`xeas`:

    from lighthouse.pipeline.xegpu.xeas import xeas

    blob = xeas(outlined_mlir)
"""

import sys

from mlir import ir
from mlir.dialects import gpu

from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import xegpu_to_binary


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
        params: Retained for compatibility with the original API. The active
            binary-only lowering does not consume these values.
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

        if assume_in_bounds:
            marked = _mark_transfers_in_bounds(module)
            if marked:
                print(
                    f"xeas: marked {marked} vector transfer op(s) as in_bounds",
                    file=sys.stderr,
                )

        schedules = [
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
    (``mgpuModuleLoad``). :func:`lighthouse.execution.xegpu.xelaunch.xelaunch`
    launches the kernel directly from it.

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

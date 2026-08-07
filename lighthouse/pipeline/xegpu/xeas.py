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

Lowering parameters that cannot be recovered from the IR are passed in via
``params`` (e.g. the matmul sizes ``m``/``n``/``k`` and any XeGPU tile
overrides); matmul sizes are inferred from the kernel signature when omitted.

The high-level entry point is :func:`xeas`:

    from lighthouse.pipeline.xegpu.xeas import xeas

    blob = xeas(outlined_mlir, params={"m": 2048, "n": 4096, "k": 8192})
"""

import sys
from itertools import product

from mlir import ir
from mlir.dialects import func, gpu

from lighthouse.pipeline.driver import TransformDriver
from lighthouse.schedule.xegpu import (
    ELEMWISE_SCHEDULE,
    MLP_SCHEDULE,
    XeGPUParameterSelector,
    xegpu_to_binary,
)
from lighthouse.schedule.xegpu.matmul_constraints import (
    DPAS,
    MIN_NB_THREADS,
    check_k_tile,
    check_load_tile_a,
    check_load_tile_b,
    check_prefetch_tile_a,
    check_prefetch_tile_b,
    check_sg_tile,
    check_wg_tile,
)
from lighthouse.schedule.xegpu.matmul_costmodel import (
    generate_load_tiles_a,
    generate_load_tiles_b,
    generate_prefetch_tiles,
)

# The XeGPU schedule locates the payload module by matching a top-level
# ``func.func`` by name. The outlined input has none (only a ``gpu.func``), so a
# throwaway function under this name is injected for the schedule to anchor on.
_SCHEDULE_ANCHOR = "__xeas_schedule_anchor"

# Tile parameters consumed by the XeGPU WG annotation stage. They form a
# dependency chain -- the load tiles are constrained by the subgroup tile and
# ``k_tile``, the prefetch tiles by the work-group tile and ``k_tile`` -- so a
# partial override cannot simply be merged into a configuration picked by the
# selector. ``_complete_tile_params`` fills the missing ones instead.
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

# Subgroup tile candidates, tried in this order when the subgroup tile has to be
# derived from a caller-provided work-group tile.
_SG_TILE_OPTIONS = (128, 64, 32, 16)


def _is_valid(check, *args, **kwargs) -> bool:
    """Report whether a constraint checker accepts its arguments."""
    try:
        check(*args, **kwargs)
    except ValueError:
        return False
    return True


def _resolve_tile(
    names: tuple[str, ...], params: dict, reference: dict, is_valid, candidates
) -> tuple[int, ...]:
    """Determine one group of interdependent tile parameters.

    A group the caller specified in full is honoured verbatim, so an explicit
    request is never silently replaced. Otherwise the missing components are
    taken from ``reference`` (the selector's configuration) when the result
    satisfies the constraints, and are derived from ``candidates`` when it does
    not. This is what happens when the caller overrode a parameter the group
    depends on.
    """
    given = {name: params[name] for name in names if name in params}
    if len(given) == len(names):
        return tuple(params[name] for name in names)

    preferred = tuple(params.get(name, reference.get(name)) for name in names)
    if None not in preferred and is_valid(preferred):
        return preferred

    for candidate in candidates():
        if all(
            name not in given or given[name] == value
            for name, value in zip(names, candidate)
        ):
            return tuple(candidate)

    raise ValueError(
        f"xeas: no valid {'/'.join(names)} exists for the given tile parameters"
        + (f" (with {given})" if given else "")
    )


def _ordered_load_tiles(tiles: list, preferred: tuple) -> list:
    """Order load tile candidates so the DPAS-shaped tile is tried first."""
    preferred = tuple(preferred)
    tiles = [tuple(tile) for tile in tiles]
    return ([preferred] if preferred in tiles else []) + [
        tile for tile in tiles if tile != preferred
    ]


def _check_wg_and_k_tile(params: dict) -> None:
    """Reject work-group / reduction tiles the lowering cannot handle.

    These head the dependency chain and are used verbatim when provided. An
    invalid one is not caught by the schedule's own assertions and instead
    surfaces much later as an opaque transform failure -- most notably
    ``k_tile == k``, which makes the reduction loop single-trip so it is
    canonicalized away and the schedule loses its handle on it.
    """
    try:
        check_wg_tile(params["m"], params["n"], (params["wg_m"], params["wg_n"]))
        check_k_tile(params["k"], params["k_tile"])
    except ValueError as error:
        raise ValueError(
            f"xeas: invalid tile parameters for matmul "
            f"{params['m']}x{params['n']}x{params['k']}: {error}"
        ) from error


def _complete_tile_params(params: dict) -> dict:
    """Fill in the tile parameters the caller did not provide.

    Starts from the caller's values and completes the configuration in
    dependency order, re-deriving anything the caller's overrides invalidate.
    """
    selector = XeGPUParameterSelector(device=params.get("device"))
    gpu_specs = selector.gpu_specs
    reference = selector.get_parameters(
        (params["m"], params["n"], params["k"]),
        params["transpose_a"],
        params["transpose_b"],
    )
    transpose_a = params["transpose_a"]
    transpose_b = params["transpose_b"]

    filled = dict(params)
    # The work-group and reduction tiles head the chain; nothing constrains them
    # beyond the problem sizes, so the selector's values can be used directly.
    for name in ("wg_m", "wg_n", "k_tile"):
        filled.setdefault(name, reference[name])
    wg_tile = (filled["wg_m"], filled["wg_n"])
    k_tile = filled["k_tile"]

    sg_tile = _resolve_tile(
        ("sg_m", "sg_n"),
        params,
        reference,
        lambda tile: _is_valid(
            check_sg_tile, wg_tile, tile, gpu_specs, min_nb_threads=MIN_NB_THREADS
        ),
        lambda: (
            tile
            for tile in product(_SG_TILE_OPTIONS, repeat=2)
            if _is_valid(
                check_sg_tile, wg_tile, tile, gpu_specs, min_nb_threads=MIN_NB_THREADS
            )
        ),
    )
    filled["sg_m"], filled["sg_n"] = sg_tile

    filled["load_a_m"], filled["load_a_k"] = _resolve_tile(
        ("load_a_m", "load_a_k"),
        params,
        reference,
        lambda tile: _is_valid(
            check_load_tile_a, tile, sg_tile, k_tile, transpose=transpose_a
        ),
        lambda: _ordered_load_tiles(
            generate_load_tiles_a(sg_tile, k_tile), DPAS.A_TILE
        ),
    )
    filled["load_b_k"], filled["load_b_n"] = _resolve_tile(
        ("load_b_k", "load_b_n"),
        params,
        reference,
        lambda tile: _is_valid(
            check_load_tile_b, tile, sg_tile, k_tile, transpose=transpose_b
        ),
        lambda: _ordered_load_tiles(
            generate_load_tiles_b(sg_tile, k_tile), DPAS.B_TILE
        ),
    )

    # Prefetch tiles are cooperative across the work group, hence they depend on
    # the work-group tile rather than the subgroup tile.
    prefetch_a, prefetch_b = generate_prefetch_tiles(
        wg_tile,
        k_tile,
        gpu_specs,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )
    filled["prefetch_a_m"], filled["prefetch_a_k"] = _resolve_tile(
        ("prefetch_a_m", "prefetch_a_k"),
        params,
        reference,
        lambda tile: _is_valid(
            check_prefetch_tile_a,
            tile,
            wg_tile,
            k_tile,
            gpu_specs,
            transpose=transpose_a,
            min_nb_threads=MIN_NB_THREADS,
        ),
        lambda: prefetch_a,
    )
    filled["prefetch_b_k"], filled["prefetch_b_n"] = _resolve_tile(
        ("prefetch_b_k", "prefetch_b_n"),
        params,
        reference,
        lambda tile: _is_valid(
            check_prefetch_tile_b,
            tile,
            wg_tile,
            k_tile,
            gpu_specs,
            transpose=transpose_b,
            min_nb_threads=MIN_NB_THREADS,
        ),
        lambda: prefetch_b,
    )

    # Prefetch depth does not interact with the tile shapes.
    for name in ("prefetch_a_nb", "prefetch_b_nb"):
        filled.setdefault(name, reference.get(name, 1))

    return filled


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


def _count_ops(root, op_name: str) -> int:
    """Return how many ops named ``op_name`` ``root`` contains, at any depth."""
    count = 0
    for region in root.operation.regions:
        for block in region.blocks:
            for op in block.operations:
                if op.operation.name == op_name:
                    count += 1
                count += _count_ops(op, op_name)
    return count


def _detect_schedule_kind(gpu_funcs: list) -> str:
    """Infer the schedule kind from the outlined kernel body.

    A single ``vector.contract`` in the kernel identifies a matmul/MLP payload;
    otherwise it is treated as an elementwise payload. The MLP schedule anchors
    on *the* DPAS op and takes one set of tile parameters, so a payload with
    several contractions (e.g. fused attention) cannot be described by it.
    """
    contractions = sum(
        _count_ops(gpu_func, "vector.contract") for gpu_func in gpu_funcs
    )
    if contractions == 1:
        return MLP_SCHEDULE
    if contractions > 1:
        print(
            f"xeas: {contractions} vector.contract ops found; the MLP schedule "
            "supports a single contraction, falling back to the elementwise "
            "schedule",
            file=sys.stderr,
        )
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
    kernel signature. A partial set of tile parameters is completed in
    dependency order, so the caller's values are preserved and everything they
    constrain is derived from them.
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

    if not all(name in params for name in _TILE_PARAM_NAMES):
        params = _complete_tile_params(params)
    _check_wg_and_k_tile(params)
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

        # schedule_kind = _detect_schedule_kind(gpu_funcs)
        # if schedule_kind == MLP_SCHEDULE:
        #     params = _prepare_mlp_params(gpu_funcs, params)

        if assume_in_bounds:
            marked = _mark_transfers_in_bounds(module)
            if marked:
                print(
                    f"xeas: marked {marked} vector transfer op(s) as in_bounds",
                    file=sys.stderr,
                )

        # anchor = _inject_schedule_anchor(module)

        schedules = [
            # build_payload_schedule(
            #     schedule_kind,
            #     [params],
            #     payload_func_name=anchor,
            #     start_at_stage="outlined",
            # ),
            xegpu_to_binary(
                xegpu_op_level=xegpu_op_level,
                large_register_file=large_register_file,
            ),
        ]
        # print(module)
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

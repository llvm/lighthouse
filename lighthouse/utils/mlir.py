"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import arith, func, linalg
import os
import platform
from pathlib import Path

_SHARED_EXT = ".dylib" if platform.system() == "Darwin" else ".so"


def get_mlir_library_path():
    """Return MLIR shared library path."""
    pkg_path = Path(ir.__file__).parent
    run_utils_lib = f"libmlir_runner_utils{_SHARED_EXT}"
    err_msg = f"Could not find shared libs in locations relative to '{pkg_path}'"
    if "python_packages" in str(pkg_path):
        # looks like a local llvm install
        try:
            # LLVM_INSTALL_DIR/python_packages/mlir_core/mlir
            # lib location: LLVM_INSTALL_DIR/lib/
            path = pkg_path.parent.parent.parent / "lib"
            assert os.path.isfile(path / run_utils_lib)
        except AssertionError:
            try:
                # LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/mlir
                # lib location: LLVM_BUILD_DIR/lib/
                path = pkg_path.parent.parent.parent.parent.parent / "lib"
                assert os.path.isfile(path / run_utils_lib)
            except AssertionError:
                raise ValueError(err_msg)
    else:
        # maybe installed in python path
        path = pkg_path / "_mlir_libs"
        assert os.path.isfile(path / run_utils_lib), err_msg
    return path


def func_cif(*args, **kwargs):
    """Like ``@func.func`` and automatically sets ``llvm.emit_c_interface``."""

    def wrap(fn):
        r = func.func(*args, **kwargs)(fn)
        r.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return r

    return wrap


def inspect_payload(payload_module: ir.Module) -> dict:
    """
    Inspect the payload module and extract metadata about the functions/ops it contains.

    Returns a dictionary:
    {
        function_name: {
            "inputs": [input types],
            "results": [result types],
            "layers": [
                {
                    "kind": "matmul",
                    "shape": (m, n, k),
                    "transpose_a": bool,
                    "transpose_b": bool,
                    ...
                },
                ...
            ]
        },
        ...
    }

    The layer list preserves walk order.
    """

    def has_producer(value: ir.Value, kind: type) -> bool:
        if value is None or isinstance(value, ir.BlockArgument):
            # stop trace
            return False
        if isinstance(value, ir.OpResult):
            parent_op = value.owner
            if isinstance(parent_op, kind):
                return True
            # recursively check producers
            for operand in parent_op.operands:
                if has_producer(operand, kind):
                    return True
        return False

    functions = {}

    def match_funcs(op: ir.Operation) -> ir.WalkResult:
        op = op.opview
        match op:
            case func.FuncOp():
                layers = []

                def match_linalg(op: ir.Operation) -> ir.WalkResult:
                    op = op.opview
                    match op:
                        # linalg.ElementwiseOp is shadowed in mlir.dialects.linalg
                        # and won't match via class pattern; match by op name.
                        case _ if op.operation.name == "linalg.elementwise":
                            outputs = op.outputs
                            assert len(outputs) == 1, "Expected only one output"
                            layers.append(
                                {
                                    "kind": "elemwise",
                                    "shape": outputs[0].type.shape,
                                    "elemtype": str(outputs[0].type.element_type),
                                }
                            )
                        case linalg.GenericOp():
                            iter_parallel = "#linalg.iterator_type<parallel>"
                            all_parallel = all(
                                str(it) == iter_parallel for it in op.iterator_types
                            )
                            inputs = op.inputs
                            outputs = op.outputs
                            if all_parallel:
                                assert len(outputs) == 1, "Expected only one output"
                                layers.append(
                                    {
                                        "kind": "elemwise",
                                        "shape": outputs[0].type.shape,
                                        "elemtype": str(outputs[0].type.element_type),
                                    }
                                )
                            else:
                                iterators = [
                                    "parallel"
                                    if str(it) == iter_parallel
                                    else "reduction"
                                    for it in op.iterator_types
                                ]
                                in_shapes = [inp.type.shape for inp in inputs]
                                out_shapes = [out.type.shape for out in outputs]
                                in_types = [
                                    str(inp.type.element_type) for inp in inputs
                                ]
                                out_types = [
                                    str(out.type.element_type) for out in outputs
                                ]
                                layers.append(
                                    {
                                        "kind": "reduction",
                                        "input_shape": in_shapes,
                                        "input_elemtype": in_types,
                                        "output_shape": out_shapes,
                                        "output_elemtype": out_types,
                                        "iterators": iterators,
                                    }
                                )
                        case linalg.MatmulOp():
                            inputs = op.inputs
                            outputs = op.outputs
                            assert len(inputs) == 2 and len(outputs) == 1
                            input_is_transpose = [
                                has_producer(o, linalg.TransposeOp) for o in inputs
                            ]
                            a_shape, b_shape = [d.type.shape for d in inputs]
                            c_shape = outputs[0].type.shape
                            assert len(c_shape) == 2
                            assert len(a_shape) == 2 or len(b_shape) == 2
                            m, n = c_shape
                            try:
                                _, k = a_shape
                            except Exception:
                                k, _ = b_shape
                            a_etype, b_etype = [
                                str(d.type.element_type) for d in inputs
                            ]
                            assert a_etype == b_etype, "Input element types must match"
                            ab_etype = a_etype
                            acc_etype = str(outputs[0].type.element_type)
                            layers.append(
                                {
                                    "kind": "matmul",
                                    "shape": (m, n, k),
                                    "ab_elemtype": ab_etype,
                                    "acc_elemtype": acc_etype,
                                    "transpose_a": input_is_transpose[0],
                                    "transpose_b": input_is_transpose[1],
                                }
                            )
                        case linalg.BatchMatmulOp():
                            inputs = op.inputs
                            outputs = op.outputs
                            assert len(inputs) == 2 and len(outputs) == 1
                            input_is_transpose = [
                                has_producer(o, linalg.TransposeOp) for o in inputs
                            ]
                            a_shape, b_shape = [d.type.shape for d in inputs]
                            c_shape = outputs[0].type.shape
                            assert len(c_shape) == 3
                            assert len(a_shape) == 3 or len(b_shape) == 3
                            b, m, n = c_shape
                            try:
                                _, _, k = a_shape
                            except Exception:
                                _, k, _ = b_shape
                            a_etype, b_etype = [
                                str(d.type.element_type) for d in inputs
                            ]
                            assert a_etype == b_etype, "Input element types must match"
                            ab_etype = a_etype
                            acc_etype = str(outputs[0].type.element_type)
                            layers.append(
                                {
                                    "kind": "batch_matmul",
                                    "shape": (b, m, n, k),
                                    "ab_elemtype": ab_etype,
                                    "acc_elemtype": acc_etype,
                                    "transpose_a": input_is_transpose[0],
                                    "transpose_b": input_is_transpose[1],
                                }
                            )
                    return ir.WalkResult.ADVANCE

                op.walk(match_linalg, ir.WalkOrder.PRE_ORDER)
                functions[op.sym_name.value] = {
                    "inputs": op.type.inputs,
                    "results": op.type.results,
                    "layers": layers,
                }
        return ir.WalkResult.ADVANCE

    for op in payload_module.body.operations:
        op.walk(match_funcs, ir.WalkOrder.PRE_ORDER)
    return functions


def opview(op: ir.Operation | ir.OpView) -> ir.OpView:
    """Return the ``OpView`` of an operation."""
    return op.opview if isinstance(op, ir.Operation) else op


def dim_position(expr: ir.AffineExpr) -> int | None:
    """Return the dimension position of a plain dimension expression.

    Returns None for non-dimension expressions (constants, composite exprs).
    """
    if isinstance(expr, ir.AffineDimExpr):
        return expr.position
    return None


def indexing_maps(op: ir.Operation | ir.OpView) -> list[ir.AffineMap] | None:
    """Return the indexing maps of a structured linalg op as ``AffineMap``s.

    The returned list follows the operand order: inputs first, then outputs.
    Returns None if the op is not a structured linalg op.
    """
    try:
        raw_maps = linalg.get_indexing_maps(opview(op))
    except (TypeError, ValueError):
        return None
    if not raw_maps:
        return None

    maps = []
    for m in raw_maps:
        maps.append(m.value if isinstance(m, ir.AffineMapAttr) else m)
    return maps


def num_loops(op: ir.Operation | ir.OpView) -> int | None:
    """Number of iteration dims (loops) of a structured linalg op, or None."""
    maps = indexing_maps(op)
    if not maps:
        return None
    return maps[0].n_dims


def linalg_inputs(op: ir.Operation | ir.OpView) -> list[ir.Value] | None:
    """Return the input (``ins``) operands of a structured linalg op.

    Works for every structured linalg op, including named ops (broadcast,
    transpose, ...) that do not expose the ``.inputs`` accessor: the operands of
    a structured linalg op are its inputs followed by its outputs (DPS inits),
    and each output is tied to one result, so the inputs are the leading
    operands. Returns None when the op is not a structured linalg op.
    """
    ov = opview(op)
    if indexing_maps(ov) is None:
        return None
    operands = list(ov.operands)
    return operands[: len(operands) - len(list(ov.results))]


def linalg_outputs(op: ir.Operation | ir.OpView) -> list[ir.Value] | None:
    """Return the output (``outs`` / init) operands of a structured linalg op.

    Works for every structured linalg op, including named ops (broadcast,
    transpose, ...) that do not expose the ``.outputs`` accessor. The outputs are
    the trailing operands, one per result (see `linalg_inputs`). Returns None
    when the op is not a structured linalg op.
    """
    ov = opview(op)
    if indexing_maps(ov) is None:
        return None
    operands = list(ov.operands)
    return operands[len(operands) - len(list(ov.results)) :]


def iterator_types(op: ir.Operation | ir.OpView) -> list[str]:
    """Iterator types of a structured linalg op as ``"parallel"``/``"reduction"``.

    A `linalg.generic` carries them as ``#linalg.iterator_type<...>`` attrs, which
    are compared against the built enum attr rather than parsed.

    Named ops (``linalg.matmul``, ``linalg.batch_matmul``, ...) have no such
    attribute, so their loop kinds are recovered from the indexing maps: a loop dim
    is parallel iff some init map references it, a reduction otherwise. That is the
    structured-op definition, and it holds for every named op whose maps are
    projected permutations.
    """
    ov = opview(op)
    if "iterator_types" in ov.operation.attributes:
        build = ir.AttrBuilder.get("linalg.IteratorTypeEnum")
        parallel = build(linalg.IteratorType.parallel, context=ov.context)
        return [
            "parallel" if it == parallel else "reduction" for it in ov.iterator_types
        ]
    maps = indexing_maps(ov)
    parallel_dims: set[int] = set()
    for imap in maps[len(linalg_inputs(ov)) :]:
        parallel_dims.update(
            expr.position for expr in imap.results if isinstance(expr, ir.AffineDimExpr)
        )
    return [
        "parallel" if pos in parallel_dims else "reduction"
        for pos in range(maps[0].n_dims)
    ]


def reduction_dims(op: ir.Operation | ir.OpView) -> list[int]:
    """Positions of the reduction iterators of a structured linalg op, in order.

    Stands in for ``LinalgOp::getReductionDims``, which has no binding.
    """
    return [i for i, it in enumerate(iterator_types(op)) if it == "reduction"]


def is_linalg_all_loops_parallel(op: ir.Operation | ir.OpView) -> bool:
    """Return True when all iterator types are parallel."""
    return all(it == "parallel" for it in iterator_types(op))


def is_linalg_eltwise_op(op: ir.Operation | ir.OpView) -> bool:
    """Return True if it is an elementwise linalg operation."""
    ov = opview(op)
    return ov.operation.name == "linalg.elementwise" or (
        isinstance(ov, linalg.GenericOp) and is_linalg_all_loops_parallel(ov)
    )


def op_users(value: ir.Value) -> list[ir.Operation]:
    """Return the ops that use `value`."""
    users = []
    for use in value.uses:
        owner = use.owner
        if isinstance(owner, ir.OpView):
            users.append(owner.operation)
        elif isinstance(owner, ir.Operation):
            users.append(owner)
    return users


def defining_op(value: ir.Value) -> ir.Operation | None:
    """
    Return the op defining `value`, or None if not possible
    e.g., defined by block arguments.
    """
    owner = value.owner
    if isinstance(owner, ir.OpView):
        return owner.operation
    if isinstance(owner, ir.Operation):
        return owner
    return None


#: Supported float element types with their bit widths. `f16` and `bf16` share a
#: width but not a format, so neither widens into the other.
_FLOAT_WIDTHS = (
    (ir.F16Type, 16),
    (ir.BF16Type, 16),
    (ir.F32Type, 32),
    (ir.F64Type, 64),
)


def float_width(element_type: ir.Type) -> int | None:
    """Bit width of a supported float type, else None."""
    for cls, width in _FLOAT_WIDTHS:
        if isinstance(element_type, cls):
            return width
    return None


def can_cast_float(from_type: ir.Type, to_type: ir.Type) -> bool:
    """Whether `cast_float` can convert between the two types."""
    if from_type == to_type:
        return True
    have, want = float_width(from_type), float_width(to_type)
    return have is not None and want is not None and have != want


def cast_float(value: ir.Value, element_type: ir.Type) -> ir.Value | None:
    """`value` converted to `element_type` via ``extf``/``truncf``, or unchanged.

    Returns None when there is no such conversion, i.e. for equal-width types of
    different format (``f16`` vs ``bf16``).
    """
    if value.type == element_type:
        return value
    if not can_cast_float(value.type, element_type):
        return None
    if float_width(element_type) > float_width(value.type):
        return arith.extf(element_type, value)
    return arith.truncf(element_type, value)


def remap_dims(imap: ir.AffineMap, dim_map: dict[int, int], num_dims: int):
    """Rewrite a pure dim-projection map through `dim_map`, or None if not pure.

    Renames the dims a map's results refer to, moving it into a different (usually
    larger) iteration space of `num_dims` dims. `dim_map` sends each of the map's
    own dim positions to a position in that space. The results keep their order and
    count and only the names change.

    Examples:

        dim_map = {0: 0, 1: 2}, num_dims = 3
        (d0, d1) -> (d0, d1)     ->  (d0, d1, d2) -> (d0, d2)
        (d0, d1) -> (d0)         ->  (d0, d1, d2) -> (d0)
        (d0, d1) -> (d1, d0)     ->  (d0, d1, d2) -> (d2, d0)

    Returns None if a result is not a plain dim expr, or its position is
    unmapped::

        (d0, d1) -> (d0 + d1)    ->  None   # not a plain dim expr
        (d0, d1) -> (d0, d1)     ->  None   # with dim_map = {0: 0}, d1 unmapped
    """
    results = []
    for expr in imap.results:
        if not isinstance(expr, ir.AffineDimExpr):
            return None
        if expr.position not in dim_map:
            return None
        results.append(ir.AffineDimExpr.get(dim_map[expr.position]))
    return ir.AffineMap.get(num_dims, 0, results)


def project_dims(imap: ir.AffineMap, projected: set[int]):
    """Drop `projected` dims from the map's domain, renumbering the rest.

    Shrinks the iteration space of the map by removing the specified projected dims.

    Examples:

        project {2}     (d0, d1, d2) -> (d0)         ->  (d0, d1) -> (d0)
        project {3}     (d0, d1, d2, d3) -> (d0, d1) ->  (d0, d1, d2) -> (d0, d1)
        project {1}     (d0, d1, d2) -> (d2, d0)     ->  (d0, d1) -> (d1, d0)

    The caller guarantees the map does not reference the projected dims, so the projection is
    lossless. Returns None if the map is not a pure dim projection, or does
    reference a projected dim:

        project {2}     (d0, d1, d2) -> (d2)         ->  None
    """
    renumber: dict[int, int] = {}
    next_pos = 0
    for pos in range(imap.n_dims):
        if pos in projected:
            continue
        renumber[pos] = next_pos
        next_pos += 1
    return remap_dims(imap, renumber, next_pos)

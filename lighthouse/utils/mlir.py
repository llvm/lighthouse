"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import func, linalg
import os
from pathlib import Path
from collections import defaultdict


def get_mlir_library_path():
    """Return MLIR shared library path."""
    pkg_path = Path(ir.__file__).parent
    run_utils_so = "libmlir_runner_utils.so"
    err_msg = f"Could not find shared libs in locations relative to '{pkg_path}'"
    if "python_packages" in str(pkg_path):
        # looks like a local llvm install
        try:
            # LLVM_INSTALL_DIR/python_packages/mlir_core/mlir
            # lib location: LLVM_INSTALL_DIR/lib/
            path = pkg_path.parent.parent.parent / "lib"
            assert os.path.isfile(path / run_utils_so)
        except AssertionError:
            try:
                # LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core/mlir
                # lib location: LLVM_BUILD_DIR/lib/
                path = pkg_path.parent.parent.parent.parent.parent / "lib"
                assert os.path.isfile(path / run_utils_so)
            except AssertionError:
                raise ValueError(err_msg)
    else:
        # maybe installed in python path
        path = pkg_path / "_mlir_libs"
        assert os.path.isfile(path / run_utils_so), err_msg
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
            "layers": {
                "matmul": {
                    "m": m,
                    "n": n,
                    "k": k,
                    "transpose_a": bool,
                    "transpose_b": bool,
                }
                ...
            }
        },
        ...
    }
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
                layers = defaultdict(list)

                def match_linalg(op: ir.Operation) -> ir.WalkResult:
                    op = op.opview
                    match op:
                        case linalg.GenericOp():
                            # TODO support ElementwiseOp and MapOp
                            iter_parallel = "#linalg.iterator_type<parallel>"
                            parallel = all(
                                str(it) == iter_parallel for it in op.iterator_types
                            )
                            assert parallel, (
                                "Only parallel iterators are supported in linalg.generic"
                            )
                            outputs = op.outputs
                            assert len(outputs) == 1, "Expected only one output"
                            out_shape = outputs[0].type.shape
                            layers["elemwise"].append(
                                {
                                    "shape": out_shape,
                                    "elemtype": str(outputs[0].type.element_type),
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
                            layers["matmul"].append(
                                {
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
                            layers["batch_matmul"].append(
                                {
                                    "shape": (b, m, n, k),
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

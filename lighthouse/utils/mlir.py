"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import func, linalg
import os
from pathlib import Path


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
            "matmuls": [(m, n, k), ...]  # list of matmul shapes
        },
        ...
    }
    """

    functions = {}

    def match_funcs(op: ir.Operation) -> ir.WalkResult:
        op = op.opview
        match op:
            case func.FuncOp():
                matmuls = []

                def match_linalg(op: ir.Operation) -> ir.WalkResult:
                    op = op.opview
                    match op:
                        case linalg.MatmulOp():
                            inputs = op.inputs
                            outputs = op.outputs
                            assert len(inputs) == 2 and len(outputs) == 1
                            m, k = inputs[0].type.shape
                            _, n = inputs[1].type.shape
                            matmuls.append((m, n, k))
                    return ir.WalkResult.ADVANCE

                op.walk(match_linalg, ir.WalkOrder.PRE_ORDER)
                functions[op.sym_name.value] = {
                    "inputs": op.type.inputs,
                    "results": op.type.results,
                    "matmuls": matmuls,
                }
        return ir.WalkResult.ADVANCE

    for op in payload_module.body.operations:
        op.walk(match_funcs, ir.WalkOrder.PRE_ORDER)
    return functions

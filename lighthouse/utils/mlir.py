"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import func
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

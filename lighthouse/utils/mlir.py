"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import func
import os


def get_mlir_library_path():
    """Return MLIR shared library path."""
    pkg_path = ir.__file__
    if "python_packages" in pkg_path:
        # looks like a local mlir install
        path = os.path.join(pkg_path.split("python_packages")[0], "lib")
    else:
        # maybe installed in python path
        path = os.path.join(os.path.split(pkg_path)[0], "_mlir_libs")
    assert os.path.isdir(path)
    return path


def func_cif(*args, **kwargs):
    """Like ``@func.func`` and automatically sets ``llvm.emit_c_interface``."""

    def wrap(fn):
        r = func.func(*args, **kwargs)(fn)
        r.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return r

    return wrap

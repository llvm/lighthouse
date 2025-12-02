"""
MLIR utility functions.
"""

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
import os


def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()


def get_mlir_library_path():
    pkg_path = ir.__file__
    if "python_packages" in pkg_path:
        # looks like a local mlir install
        path = pkg_path.split("python_packages")[0] + os.sep + "lib"
    else:
        # maybe installed in python path
        path = os.path.split(pkg_path)[0] + os.sep + "_mlir_libs"
    assert os.path.isdir(path)
    return path

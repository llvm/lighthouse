import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


def import_mlir_module(path: str, context: ir.Context) -> ir.Module:
    """Import an MLIR text file into an MLIR module"""
    if path is None:
        raise ValueError("Path to the module must be provided.")
    if not os.path.exists(path):
        raise ValueError(f"Path to the module does not exist: {path}")
    with open(path, "r") as f:
        return ir.Module.parse(f.read(), context=context)


@lru_cache(maxsize=None)
def _resolve_package(directory: Path) -> tuple[str, str]:
    """
    Resolve the enclosing package for a directory.

    Results are cached per directory so the walk runs at most once
    for each directory.
    """
    package_parts: list[str] = []
    parent = directory
    while (parent / "__init__.py").exists():
        package_parts.insert(0, parent.name)
        parent = parent.parent
    return str(parent), ".".join(package_parts)


def import_python_module(path: str) -> ModuleType:
    """Import a Python module from a file."""
    if path is None:
        raise ValueError("Path to the module must be provided.")
    if not os.path.exists(path):
        raise ValueError(f"Path to the module does not exist: {path}")

    filepath = Path(path).resolve()

    # Resolve the enclosing package to enable relative imports.
    package_root, dotted_prefix = _resolve_package(filepath.parent)

    module_name = filepath.stem
    if dotted_prefix:
        # The file is part of a package: build its dotted name and make sure
        # the package root is importable so relative imports resolve.
        qualified_name = f"{dotted_prefix}.{module_name}"
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        return importlib.import_module(qualified_name)

    # Standalone file: load it directly from its location.
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_registered_pass(*args, **kwargs):
    """Utility function to add a bundle of passes to a Transform Schedule"""
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    """Matches a pattern to AnyType"""
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    """Runs canonicalization patterns on the given operation"""
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()


def cleanup_func(target):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_cse(func)
    canonicalize(func)


class PipelineInterrupt(Exception):
    """Exception to signal early termination of the transform schedule."""

    pass


def match_and_split(*args, nhandles=1, **kwargs):
    """Helper function that splits matched handles."""
    matched = match(*args, **kwargs)
    anytype = transform.AnyOpType.get()
    matched_ops = transform.split_handle((anytype,) * nhandles, matched)
    if nhandles == 1:
        matched_ops = [matched_ops]
    return matched_ops

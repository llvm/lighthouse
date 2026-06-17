import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from mlir import ir


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

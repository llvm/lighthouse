import os

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

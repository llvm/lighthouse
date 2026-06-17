from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


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

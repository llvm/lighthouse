from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured


# Utility function to add a bundle of passes to a Transform Schedule. This can be used to easily add a predefined set of passes to a pipeline.
def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform
from mlir.dialects.transform import structured


class PassBundles:
    """
    Predefined pass bundles for common transformations. These are not exhaustive and can be extended as needed.
    The idea is to group together passes that are commonly used together in a pipeline, so that they can be easily added to a PassManager or Transform Schedule with a single function call.
    """

    # All in one bufferization bundle. This is self consistent and should be used together.
    BufferizationBundle = [
        "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}",
        "drop-equivalent-buffer-results",
        "buffer-deallocation-pipeline",
    ]

    # Lowers most dialects to LLVM Dialect
    LLVMLoweringBundle = [
        "convert-linalg-to-loops",
        "convert-scf-to-cf",
        "convert-to-llvm",
        "reconcile-unrealized-casts",
    ]

    # Canonicalization bundle. This is a set of passes that can be used to clean up the IR after transformations.
    CleanupBundle = [
        "cse",
        "canonicalize",
    ]


# Utility function to add a bundle of passes to a PassManager. This can be used to easily add a predefined set of passes to a pipeline.
def add_bundle(pm: PassManager, bundle: list[str]) -> None:
    for p in bundle:
        pm.add(p)


# Utility function to add a bundle of passes to a Transform Schedule. This can be used to easily add a predefined set of passes to a pipeline.
def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()

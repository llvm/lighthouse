from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform
from mlir.dialects.transform import structured


class Pass:
    """
    A simple wrapper class for MLIR passes.
    The options can be serialized into a string for consumption by the PassManager.
    Or used directly with the Transform Schedule by passing the options as a dictionary.
    """

    def __init__(self, name: str, options: dict = {}):
        self.name = name
        self.options = options

    def __str__(self) -> str:
        """serialize name + options dictionary for pass manager consumption"""
        if not self.options:
            return self.name
        options_str = " ".join(f"{key}={value}" for key, value in self.options.items())
        return f"{self.name}{{{options_str}}}"


class PassBundles:
    """
    Predefined pass bundles for common transformations. These are not exhaustive and can be extended as needed.
    The idea is to group together passes that are commonly used together in a pipeline, so that they can be easily added to a PassManager or Transform Schedule with a single function call.
    """

    # All in one bufferization bundle. This is self consistent and should be used together.
    BufferizationBundle = [
        Pass(
            "one-shot-bufferize",
            {
                "function-boundary-type-conversion": "identity-layout-map",
                "bufferize-function-boundaries": True,
            },
        ),
        Pass("drop-equivalent-buffer-results"),
        # This last pass only works with the pass manager, not schedules.
        # Pass("buffer-deallocation-pipeline"),
    ]

    # Lowers most dialects to basic control flow.
    MLIRLoweringBundle = [
        Pass("convert-linalg-to-loops"),
    ]

    # Lowers most dialects to LLVM Dialect
    LLVMLoweringBundle = [
        Pass("convert-scf-to-cf"),
        Pass("convert-to-llvm"),
        Pass("reconcile-unrealized-casts"),
    ]

    # Canonicalization bundle. This is a set of passes that can be used to clean up the IR after transformations.
    CleanupBundle = [
        Pass("cse"),
        Pass("canonicalize"),
    ]


# Utility function to add a bundle of passes to a PassManager. This can be used to easily add a predefined set of passes to a pipeline.
def add_bundle(pm: PassManager, bundle: list[Pass]) -> None:
    for p in bundle:
        pm.add(str(p))


# Utility function to add a bundle of passes to a Schedule. This can be used to easily add a predefined set of passes to a pipeline.
def apply_bundle(op, bundle: list[Pass], *args, **kwargs) -> None:
    for p in bundle:
        op = apply_registered_pass(op, p.name, options=p.options, *args, **kwargs)
    return op


# Utility function to add a bundle of passes to a Transform Schedule. This can be used to easily add a predefined set of passes to a pipeline.
def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


def match(*args, **kwargs):
    return structured.structured_match(transform.AnyOpType.get(), *args, **kwargs)


def canonicalize(op):
    with ir.InsertionPoint(transform.apply_patterns(op).patterns):
        transform.apply_patterns_canonicalization()


def cleanup_func(target):
    func = structured.MatchOp.match_op_names(target, ["func.func"]).result
    transform.apply_cse(func)
    canonicalize(func)

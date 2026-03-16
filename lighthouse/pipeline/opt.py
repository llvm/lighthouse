from abc import abstractmethod
import os

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform
from lighthouse.pipeline.helper import import_mlir_module


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


# Predefined pass bundles for common transformations.
# These are not exhaustive and can be extended as needed.
# The idea is to group together passes that are commonly used together in a pipeline,
# so that they can be easily added to a PassManager or Transform Schedule with a single function call.
PassBundles = {
    # All in one bufferization bundle.
    # This is self consistent and should be used together.
    "BufferizationBundle": [
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
    ],
    # Lowers most dialects to basic control flow.
    "MLIRLoweringBundle": [
        Pass("convert-linalg-to-loops"),
    ],
    # Lowers most dialects to LLVM Dialect
    "LLVMLoweringBundle": [
        Pass("convert-scf-to-cf"),
        Pass("convert-to-llvm"),
        Pass("reconcile-unrealized-casts"),
    ],
    # Canonicalization bundle.
    "CleanupBundle": [
        Pass("cse"),
        Pass("canonicalize"),
    ],
}


# Utility function to add a bundle of passes to a PassManager.
def add_bundle(pm: PassManager, bundle: list[Pass]) -> None:
    for p in bundle:
        pm.add(str(p))


# Utility function to add a bundle of passes to a Schedule.
def apply_bundle(op, bundle: list[Pass], *args, **kwargs) -> None:
    for p in bundle:
        op = transform.apply_registered_pass(
            transform.AnyOpType.get(), op, p.name, options=p.options, *args, **kwargs
        )
    return op


class Transform:
    """
    A simple wrapper class for MLIR transforms.
    Keeps the file name of the transform module to load,
    to be easily passed to a TransformStage.
    """

    def __init__(self, filename: str):
        self.filename = filename

    def __str__(self) -> str:
        """serialize filename for transform stage consumption"""
        return f"{self.filename}"


class Stage:
    """
    A stage in the optimization pipeline. Each stage will apply a specific set of transformations to the module,
    and will keep track of the current state of the module after the transformations are applied.
    """

    @abstractmethod
    def apply(self, module: ir.Module) -> ir.Module:
        """
        Apply the transformations for this stage to the given module, and return the transformed module.
        """
        pass


class PassStage(Stage):
    """
    A stage that applies a predefined set of passes to the module. This is a simple wrapper around a PassManager.
    """

    def __init__(self, passes: list[Pass], context: ir.Context):
        self.context = context
        self.pm = PassManager("builtin.module", self.context)
        add_bundle(self.pm, passes)

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply passes to.")
        if module.context != self.context:
            raise ValueError("Module context does not match PassManager context.")
        self.pm.run(module.operation)
        return module


class TransformStage(Stage):
    """
    A stage that applies a predefined set of transformations to the module. This is a simple wrapper around a Transform Schedule.
    """

    def __init__(self, filename: str, context: ir.Context):
        self.module = import_mlir_module(filename, context)
        if "transform.with_named_sequence" not in self.module.operation.attributes:
            raise ValueError(
                f"Transform module {filename} does is not a transform schedule."
            )
        self.schedule = self.module.body.operations[0]

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply transformations to.")
        self.schedule.apply(module.operation)
        return module


class Driver:
    """
    A simple driver that runs the optimization pipeline on a given workload.
    This is a high-level interface that abstracts away the details of the optimization pipeline,
    and provides a simple interface for running the pipeline on a given workload.

    The pipeline is flexible until the first time it is run, at which point it becomes fixed and cannot be modified until reset is called.
    This is to allow running the same pipeline on different modules, without accidentally modifying the pipeline after it has been run.

    Calling reset() will clear the pipeline and the module, allowing for a new pipeline to be constructed and run on a new module.
    """

    def __init__(self, filename: str, stages: list[str] = []):
        # The context is shared across the entire pipeline, and is used to create the PassManager and Transform Schedules.
        # The module is owned by the Driver to encapsulate its use through the pipeline.
        # It is returned at the end of run() after being transformed by the stages in the pipeline.
        self.context = ir.Context()
        self.module = None
        if filename:
            self.import_payload(filename)
        self.pipeline: list[Stage] = []
        self.pipeline_fixed = False
        self.bundles = PassBundles
        if stages:
            self.add_stages(stages)

    def import_payload(self, path: str) -> None:
        """Import the payload module and set it as the current module in the pipeline."""
        if self.module is not None:
            raise ValueError("Module already imported. Reset to start again.")
        self.module = import_mlir_module(path, self.context)

    def add_stage(self, stage_name: str) -> None:
        if self.pipeline_fixed:
            raise ValueError("Pipeline is fixed. Reset to start again.")
        # Pass Bundle
        if stage_name in self.bundles:
            self.pipeline.append(PassStage(self.bundles[stage_name], self.context))
        # Transform
        elif os.path.exists(stage_name):
            self.pipeline.append(TransformStage(stage_name, self.context))
        # Assume random strings represent a single pass
        # Will crash later if the pass name is not registered.
        else:
            self.pipeline.append(PassStage([Pass(stage_name)], self.context))

    def add_stages(self, stages: list[str]) -> None:
        for s in stages:
            self.add_stage(s)

    def reset(self) -> None:
        """Reset the pipeline to an empty state, allowing for new stages to be added."""
        self.pipeline = list[Stage]()
        self.module = None
        self.pipeline_fixed = False

    def run(self) -> ir.Module:
        if self.module is None:
            raise ValueError("Module must not be empty.")
        if len(self.pipeline) == 0:
            raise ValueError("Pipeline must have at least one stage.")
        for stage in self.pipeline:
            self.module = stage.apply(self.module)

        # The pipeline is now fixed and cannot be modified until reset is called.
        # This is to prevent accidental modifications to the pipeline after it has been run,
        # and to ensure that different pipelines are not run on different modules.
        self.pipeline_fixed = True

        # We don't want to run the pipeline twice on the same module,
        # so we clear the module from the driver after running the pipeline,
        # and return it to the caller.
        # To use the pipeline again, the caller must import a new module into the driver.
        module = self.module
        self.module = None
        return module

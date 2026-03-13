from abc import abstractmethod

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform


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


# Predefined pass bundles for common transformations. These are not exhaustive and can be extended as needed.
# The idea is to group together passes that are commonly used together in a pipeline, so that they can be easily added to a PassManager or Transform Schedule with a single function call.
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


class Stage:
    """
    A stage in the optimization pipeline. Each stage will apply a specific set of transformations to the module,
    and will keep track of the current state of the module after the transformations are applied.
    """

    def __init__(self, name: str):
        self.name = name

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

    def __init__(self, name: str, context: ir.Context, passes: list[Pass]):
        super().__init__(name)
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

    def __init__(self, name: str, schedule: transform.TransformOpInterface):
        super().__init__(name)
        self.schedule = schedule

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply transformations to.")
        self.schedule.apply(module.operation)
        return module


class Opt:
    """
    A simple optimizing pipeline that applies a predefined set of passes to an MLIR module.

    The module will be changed in-place throughout the pipeline. Running passes or transform schedules
    will change the internal representation, and this class will keep track of the current state of the module.
    """

    def __init__(self, payload_module: ir.Module):
        self.payload_module = payload_module
        if self.payload_module is None:
            raise ValueError("Payload module must not be empty.")
        self.pipeline = list[Stage]()

    def add_stage(self, stage: Stage) -> None:
        if isinstance(stage, Stage):
            self.pipeline.append(stage)
        elif isinstance(stage, list) and all(isinstance(s, Stage) for s in stage):
            for s in stage:
                self.pipeline.append(s)
        elif isinstance(stage, list) and all(isinstance(s, Pass) for s in stage):
            for p in stage:
                self.pipeline.append(PassStage(p, self.payload_module.context, [p]))
        else:
            raise ValueError(
                "Stage must be an instance of Stage or a list of Stage, or a list of Pass instances."
            )

    def apply(self) -> ir.Module:
        if self.payload_module is None:
            raise ValueError("Twice apply won't fly.")
        for stage in self.pipeline:
            self.payload_module = stage.apply(self.payload_module)
        # Nullify the module to prevent accidental reuse.
        ret = self.payload_module
        self.payload_module = None
        return ret


class Driver:
    """
    A simple driver that runs the optimization pipeline on a given workload.
    This is a high-level interface that abstracts away the details of the optimization pipeline,
    and provides a simple interface for running the pipeline on a given workload.

    Note: this driver is opinionated and incomplete. For now it only adds some bundles,
    no schedules, no custom passes. This will change in time.
    """

    def __init__(self, module: ir.Module):
        self.context = module.context
        self.opt = Opt(module)
        self.available_stages = PassBundles

    def add_stage(self, stage_name: str) -> None:
        # If not registered, add the stage as a pass stage with the given name.
        # This allows for ad-hoc stages to be added to the pipeline,
        # but also ensures that registered stages are added with the correct passes.
        if stage_name in self.available_stages:
            self.opt.add_stage(self.available_stages[stage_name])
        else:
            self.opt.add_stage(PassStage(stage_name, self.context, [Pass(stage_name)]))

    def run(self) -> ir.Module:
        return self.opt.apply()

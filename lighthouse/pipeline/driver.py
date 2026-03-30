import os

from mlir import ir
import lighthouse.pipeline.stage as lhs
from lighthouse.pipeline.helper import import_mlir_module, remove_args_and_opts
from lighthouse.pipeline.descriptor import PipelineDescriptor


class PipelineDriver:
    """
    A simple driver that runs the optimization pipeline on a given workload.
    Helps create a list of Stages (passes, transforms, bundles) to apply to the module, and runs them in sequence.
    """

    stages: list[lhs.Stage]
    context: ir.Context

    def __init__(self, context: ir.Context):
        self.context = context
        self.stages = []

    def add_pass(self, name: str) -> None:
        # Assume the pass name exists, will crash later if it does not.
        self.stages.append(lhs.PassStage([lhs.Pass(name)], self.context))

    def add_transform(self, stage: str | ir.Module) -> None:
        # Transform will figure out if this is MLIR, Python or Module, and will handle it accordingly.
        if isinstance(stage, ir.Module):
            # This is a transform already in module form. Assume it has been verified already.
            if stage.context != self.context:
                raise ValueError("Module context does not match driver context.")
            self.stages.append(lhs.TransformStage(stage, self.context))
        elif isinstance(stage, str):
            self.stages.append(lhs.TransformStage(lhs.Transform(stage), self.context))
        else:
            raise ValueError(f"Unsupported stage type: {type(stage)}")

    def add_bundle(self, name: str) -> None:
        # A bundle name that must exist.
        if name not in lhs.PassBundles:
            raise ValueError(f"Unknown pass bundle: {name}")
        self.stages.append(lhs.PassStage(lhs.PassBundles[name], self.context))

    def add_stage(self, stage: lhs.Stage) -> None:
        # A generit stage that isn't covered by the existing infrastructure.
        # Users can derive their own classes from Stage and add them to the pipeline with this method.
        self.stages.append(stage)

    def apply(self, module: ir.Module) -> ir.Module:
        if module.context != self.context:
            raise ValueError("Module context does not match driver context.")
        for stage in self.stages:
            module = stage.apply(module)
        return module

    def __len__(self):
        return len(self.stages)

    def reset(self):
        self.stages = []


class TransformDriver(PipelineDriver):
    """
    A simple driver that runs a sequence of transform modules on a given workload.
    This is a thin wrapper around PipelineDriver that is used to run a sequence of transform modules on a given workload.
    """

    def __init__(self, schedules: list[ir.Module]):
        if not isinstance(schedules, list) or not all(
            isinstance(s, ir.Module) for s in schedules
        ):
            raise ValueError("Schedules must be a list of ir.Module")
        super().__init__(schedules[0].context)

        for s in schedules:
            self.add_transform(s)


class CompilerDriver:
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
        self.pipeline = PipelineDriver(self.context)
        self.pipeline_fixed = False
        self.bundles = lhs.PassBundles
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

        # Stages can contain arguments and options, clean up for os checks
        filename = remove_args_and_opts(stage_name)

        if stage_name in self.bundles:
            # Pass Bundle
            self.pipeline.add_bundle(stage_name)

        elif os.path.exists(filename):
            # Transform or YAML
            if filename.endswith(".mlir") or filename.endswith(".py"):
                self.pipeline.add_transform(stage_name)
            elif filename.endswith(".yaml"):
                desc = PipelineDescriptor(stage_name)
                for s in desc.get_stages():
                    self.add_stage(s)
            else:
                _, ext = os.path.splitext(filename)
                raise ValueError(f"Unknown file type '{ext}' for stage '{stage_name}'.")

        else:
            # Assume random strings represent a single pass
            # Will crash later if the pass name is not registered.
            self.pipeline.add_pass(stage_name)

    def add_stages(self, stages: list[str]) -> None:
        for s in stages:
            self.add_stage(s)

    def reset(self) -> None:
        """Reset the pipeline to an empty state, allowing for new stages to be added."""
        self.pipeline.reset()
        self.module = None
        self.pipeline_fixed = False

    def run(self) -> ir.Module:
        if self.module is None:
            raise ValueError("Module must not be empty.")
        if len(self.pipeline) == 0:
            raise ValueError("Pipeline must have at least one stage.")

        # Apply the whole pipeline.
        self.pipeline.apply(self.module)

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

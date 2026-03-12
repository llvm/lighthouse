from abc import abstractmethod

from mlir import ir
from mlir.passmanager import PassManager

from lighthouse.pipeline.helper import PassBundles, add_bundle
from mlir.dialects import transform


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

    def __init__(self, name: str, context: ir.Context, passes: list[str]):
        super().__init__(name)
        self.pm = PassManager("builtin.module", context)
        add_bundle(self.pm, passes)

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply passes to.")
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
        self.pipeline.append(stage)

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
    """

    def __init__(self, module: ir.Module):
        self.context = module.context
        self.opt = Opt(module)

    def bufferize(self) -> None:
        pipeline = PassBundles.BufferizationBundle + PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("Bufferization", self.context, pipeline))

    def mlir_lowering(self) -> None:
        pipeline = PassBundles.MLIRLoweringBundle + PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("MLIR Lowering", self.context, pipeline))

    def llvm_lowering(self) -> None:
        pipeline = PassBundles.LLVMLoweringBundle + PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("LLVM Lowering", self.context, pipeline))

    def cleanup(self) -> None:
        pipeline = PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("Cleanup", self.context, pipeline))

    def run(self) -> ir.Module:
        return self.opt.apply()

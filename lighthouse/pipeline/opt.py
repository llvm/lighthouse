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

    def __init__(self, name: str, module: ir.Module):
        self.name = name
        self.module = module

    @abstractmethod
    def apply(self) -> ir.Module:
        """
        Apply the transformations for this stage to the given module, and return the transformed module.
        """
        pass


class PassStage(Stage):
    """
    A stage that applies a predefined set of passes to the module. This is a simple wrapper around a PassManager.
    """

    def __init__(self, name: str, module: ir.Module, passes: list[str]):
        super().__init__(name, module)
        self.pm = PassManager("builtin.module", self.module.context)
        add_bundle(self.pm, passes)

    def apply(self) -> ir.Module:
        if self.module is None:
            raise ValueError("Twice apply won't fly.")
        self.pm.run(self.module.operation)
        # Nullify the module to prevent accidental reuse.
        ret = self.module
        self.module = None
        return ret


class TransformStage(Stage):
    """
    A stage that applies a predefined set of transformations to the module. This is a simple wrapper around a Transform Schedule.
    """

    def __init__(
        self, name: str, module: ir.Module, schedule: transform.TransformOpInterface
    ):
        super().__init__(name, module)
        self.schedule = schedule

    def apply(self) -> ir.Module:
        if self.module is None:
            raise ValueError("Twice apply won't fly.")
        self.schedule.apply(self.module.operation)
        # Nullify the module to prevent accidental reuse.
        ret = self.module
        self.module = None
        return ret


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
            self.payload_module = stage.apply()
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
        self.module = module
        self.opt = Opt(self.module)

    def bufferize(self) -> None:
        bufferization_pipeline = (
            PassBundles.BufferizationBundle + PassBundles.CleanupBundle
        )
        self.opt.add_stage(
            PassStage("Bufferization", self.module, bufferization_pipeline)
        )

    def mlir_lowering(self) -> None:
        lowering_pipeline = PassBundles.MLIRLoweringBundle + PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("MLIR Lowering", self.module, lowering_pipeline))

    def llvm_lowering(self) -> None:
        lowering_pipeline = PassBundles.LLVMLoweringBundle + PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("LLVM Lowering", self.module, lowering_pipeline))

    def cleanup(self) -> None:
        bufferization_pipeline = PassBundles.CleanupBundle
        self.opt.add_stage(PassStage("Cleanup", self.module, bufferization_pipeline))

    def run(self) -> ir.Module:
        return self.opt.apply()

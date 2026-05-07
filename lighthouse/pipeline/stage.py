from abc import abstractmethod
import importlib
from enum import Enum
from pathlib import Path
import os

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform
from lighthouse.pipeline.helper import import_mlir_module
from lighthouse.pipeline.descriptor import Descriptor, PipelineDescriptor


class Pass:
    """
    A simple wrapper class for MLIR passes.
    The options can be serialized into a string for consumption by the PassManager.
    Or used directly with the Transform Schedule by passing the options as a dictionary.
    """

    def __init__(self, desc: Descriptor):
        if not isinstance(desc, Descriptor):
            raise ValueError(
                f"Pass must be initialized with a Descriptor, got {type(desc)}"
            )
        self.name = desc.basename
        self.options = desc.opts

    def __str__(self) -> str:
        """serialize name + options dictionary for pass manager consumption"""
        if not self.options:
            return self.name
        options_str = " ".join(f"{key}={value}" for key, value in self.options.items())
        return f"{self.name}{{{options_str}}}"


# Utility function to add a bundle of passes to a PassManager.
def add_bundle(pm: PassManager, filename: str) -> None:
    desc = PipelineDescriptor(Descriptor(filename))
    for p in desc.get_stages():
        if not isinstance(p, Descriptor) or not p.is_pass():
            raise ValueError(
                f"Expected Descriptor of type Pass in bundle, got {type(p)}"
            )
        pm.add(str(Pass(p)))


# Utility function to add a bundle of passes to a Schedule.
def apply_bundle(op, filename: str, *args, **kwargs) -> None:
    desc = PipelineDescriptor(Descriptor(filename))
    for p in desc.get_stages():
        if not isinstance(p, Descriptor) or not p.is_pass():
            raise ValueError(
                f"Expected Descriptor of type Pass in bundle, got {type(p)}"
            )
        op = transform.apply_registered_pass(
            transform.AnyOpType.get(), op, p.basename, options=p.opts, *args, **kwargs
        )
    return op


class Transform:
    """
    A simple wrapper class for MLIR transforms.
    Keeps the file name of the transform module to load,
    to be easily passed to a TransformStage.

    Arguments:
      * filename: the file that will be imported into a schedule (mlir or python)

    In the filename, the arguments ([...]) will define:
      * gen: function name in case of a python file,
             what name to look for to get the MLIR module
      * seq: the named sequence to look for. FIXME: This is not implemented yet.
             Empty will pick the first.

    In the filename, the options ({...}) will be stored as a dict
    and can be passed to the gen function
    """

    class Type(Enum):
        MLIR = 1
        Python = 2

    def __init__(self, desc: Descriptor):
        if not isinstance(desc, Descriptor):
            raise ValueError(
                f"Transform must be initialized with a Descriptor, got {type(desc)}"
            )
        if desc.basename.endswith(".mlir"):
            self.type = Transform.Type.MLIR
        elif desc.basename.endswith(".py"):
            self.type = Transform.Type.Python
        else:
            raise ValueError(f"Unsupported transform file type: {desc.basename}")
        self.filename = desc.basename
        self.options = desc.opts
        self.generator = desc.args["gen"] if "gen" in desc.args else "create_schedule"
        self.sequence = desc.args["seq"] if "seq" in desc.args else ""

    def __str__(self) -> str:
        """serialize name + filename for debugging purposes"""
        if not self.options:
            return self.filename
        return f"{self.filename}{{{self.options}}}"


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
        self.passes = passes
        for p in passes:
            self.pm.add(str(p))

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply passes to.")
        if module.context != self.context:
            raise ValueError("Module context does not match PassManager context.")
        self.pm.run(module.operation)
        return module

    def __str__(self) -> str:
        return f"PassStage({[str(p) for p in self.passes]})"


class TransformStage(Stage):
    """
    A stage that applies a predefined set of transformations to the module.
    This is a simple wrapper around a Transform Schedule.

    The MLIR file format must have:
      * a module with attributes {transform.with_named_sequence}
      * transform.named_sequence inside that module (for now, only the first is considered)

    The Python file format must have:
      * a function called create_schedule (or a name that is passed as argument)
      * (optional) a dictionary argument for the options
    """

    MLIR_ATTRIBUTE = "transform.with_named_sequence"

    def __init__(self, transform: Transform | ir.Module, context: ir.Context):
        if isinstance(transform, ir.Module):
            # If the module is passed directly, we expect it to be already loaded and validated.
            self.module = transform
        elif transform.type == Transform.Type.MLIR:
            # For MLIR transforms, we expect the file to define an MLIR transform sequence
            # that we can import and apply to the module. This will be checked below.
            self.module = import_mlir_module(transform.filename, context)
        elif transform.type == Transform.Type.Python:
            # For Python transforms, we expect the file to define a function
            # that takes an MLIR module and returns a transformed MLIR module.
            module_name = Path(os.path.basename(transform.filename)).stem
            spec = importlib.util.spec_from_file_location(
                module_name, transform.filename
            )
            transform_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(transform_module)
            if not hasattr(transform_module, transform.generator):
                raise ValueError(
                    f"Transform module '{transform.filename}' does not define a '{transform.generator}' generator function."
                )
            self.generator = getattr(transform_module, transform.generator)

            # Run the function with the dictionary as the options that will create the named sequence.
            with context, ir.Location.unknown():
                self.module = self.generator(transform.options)
        else:
            raise ValueError(f"Unsupported transform type: {transform.type}")

        # Check if the imported module contains at least one schedule
        if TransformStage.MLIR_ATTRIBUTE not in self.module.operation.attributes:
            raise ValueError(
                f"Transform module {transform.filename} does not define a {TransformStage.MLIR_ATTRIBUTE} attribute."
            )

        # Assume the first (or only) sequence.
        self.schedule = self.module.body.operations[0]
        # TODO: Implement a search for named sequences.

    def apply(self, module: ir.Module) -> ir.Module:
        if module is None:
            raise ValueError("Missing module to apply transformations to.")
        self.schedule.apply(module.operation)
        return module

    def __str__(self) -> str:
        return f"TransformStage({self.module})"

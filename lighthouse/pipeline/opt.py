from abc import abstractmethod
import importlib
from enum import Enum
from pathlib import Path
import os
import re

from mlir import ir
from mlir.passmanager import PassManager
from mlir.dialects import transform
from lighthouse.pipeline.helper import import_mlir_module
from lighthouse.pipeline.descriptor import PipelineDescriptor


def convert_string(value: str) -> str | int | float | bool:
    if value == "True":
        return True
    elif value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_csv(line: str, separator: str = ",") -> dict:
    result = {}
    arg_tuples = line.split(separator)
    for arg in arg_tuples:
        if not arg:
            continue
        if "=" in arg:
            key, value = arg.split("=")
            result[key] = convert_string(value)
        else:
            result[arg] = True
    return result


def remove_args_and_opts(line: str) -> str:
    if m := re.search(r"^([^[{]*)", line):
        line = m.group(0)
    return line


def parse_args_and_opts(line: str) -> tuple[str, dict, dict]:
    args = {}
    options = {}

    # Args: [arg1=val1,args2]
    if m := re.search(r"\[([^]]*)\]", line):
        args_str = m.group(1)
        args = parse_csv(args_str, ",")

    # Opts: {arg1=val1 args2}
    if m := re.search(r"\{([^}]+)\}", line):
        opts_str = m.group(1)
        options = parse_csv(opts_str, " ")

    # Cleanup the original string
    line = remove_args_and_opts(line)

    return [line, args, options]


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

    def __init__(self, filename: str):
        # First, eliminate arguments and options
        filename, args, self.options = parse_args_and_opts(filename)
        if filename.endswith(".mlir"):
            self.type = Transform.Type.MLIR
        elif filename.endswith(".py"):
            self.type = Transform.Type.Python
        else:
            raise ValueError(f"Unsupported transform file type: {filename}")
        self.filename = filename
        self.generator = args["gen"] if "gen" in args else "create_schedule"
        self.sequence = args["seq"] if "seq" in args else ""

    def __str__(self) -> str:
        """serialize name + filename for debugging purposes"""
        if not self.options:
            return self.name
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

    def __init__(self, transform: Transform, context: ir.Context):
        if transform.type == Transform.Type.MLIR:
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

        # Stages can contain arguments and options, clean up for os checks
        filename = remove_args_and_opts(stage_name)

        if stage_name in self.bundles:
            # Pass Bundle
            self.pipeline.append(PassStage(self.bundles[stage_name], self.context))

        elif os.path.exists(filename):
            # Transform or YAML
            if filename.endswith(".mlir") or filename.endswith(".py"):
                self.pipeline.append(
                    TransformStage(Transform(stage_name), self.context)
                )
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

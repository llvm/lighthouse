import yaml
import os
import re


class Descriptor:
    """
    A data class to represent a pipeline descriptor, containing:
    - The basename of the descriptor, for pass/transform identification.
    - The arguments [] to the transform, if any, in the form of a dictionary.
    - The options {} to the transform, if any, in the form of a dictionary.
    - The type of the descriptor, if specified. Can be inferred.
    - The base_path of the parent descriptor, for resolving includes.
    """

    search_path = {
        ".py": "../schedule",
        ".yaml": "./descriptors",
    }

    def __init__(
        self,
        descriptor: str = "",
        args: dict = None,
        opts: dict = None,
        type: str = None,
        base_path: str = None,
    ):
        self.type = type
        self.base_path = base_path
        if descriptor and not args and not opts:
            self.basename, self.args, self.opts = self._parse_args_and_opts(descriptor)
        else:
            self.basename = descriptor
            self.args = args if args is not None else {}
            self.opts = opts if opts is not None else {}

        # Normalize the include path if the descriptor is an include or transform
        if self.is_include() or self.is_transform():
            self.basename = self._normalize_include_path()

        # If no base_path is passed, set it to the directory of the descriptor.
        if self.base_path is None and self.basename:
            self.base_path = os.path.dirname(self.basename)

    def is_pass(self) -> bool:
        if self.type == "pass":
            return True
        if self.type in ("transform", "include"):
            return False
        # If type not passed
        pattern = re.compile(r"\.\w+$")
        if (
            self.basename
            and not self.args
            and not self.opts
            and not pattern.search(self.basename)
        ):
            self.type = "pass"
            return True
        return False

    def is_include(self) -> bool:
        if self.type == "include":
            return True
        if self.type in ("pass", "transform"):
            return False
        # If type not passed
        if (
            self.basename
            and not self.args
            and not self.opts
            and self.basename.endswith(".yaml")
        ):
            self.type = "include"
            return True
        return False

    def is_transform(self) -> bool:
        if self.type == "transform":
            return True
        if self.type in ("pass", "include"):
            return False
        # If type not passed
        if self.basename and (
            self.basename.endswith(".mlir") or self.basename.endswith(".py")
        ):
            self.type = "transform"
            return True
        return False

    def _normalize_include_path(self) -> str:
        """
        Finds the file in some standard locations, in order:
            * The path of the descriptor file that includes it. This allows for relative includes.
            * The path of the Lighthouse schedule module, where all the standard pipelines are located.
        """
        # If absolute path, check if it exists and return.
        if os.path.isabs(self.basename):
            if os.path.exists(self.basename):
                return self.basename
            else:
                raise ValueError(
                    f"Included pipeline descriptor file does not exist: {self.basename}"
                )

        # First look in the same directory as the including file, to allow for relative includes.
        filename = self._remove_args_and_opts(self.basename)
        if self.base_path and os.path.exists(self.base_path):
            file = os.path.join(self.base_path, filename)
            if os.path.exists(file):
                return file

        # If not found, look for an include path, based on the file extension.
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in self.search_path:
            raise ValueError(
                f"Included pipeline descriptor file does not exist: {filename} \
                    (searched in {self.base_path})"
            )

        # If include path, look in the descriptor/schedule module path.
        schedule_module_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), self.search_path[file_ext])
        )
        file = os.path.join(schedule_module_path, filename)
        if os.path.exists(file):
            return file

        raise ValueError(
            f"Included pipeline descriptor file does not exist: {filename} \
                (searched in {self.base_path} and {schedule_module_path})"
        )

    @staticmethod
    def _string_to_type(value: str) -> str | int | float | bool | list:
        value = str(value)
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
                # List of values, e.g. [val1,val2,...]
                if value.startswith("[") and value.endswith("]"):
                    list_str = value[1:-1]
                # List of values, e.g. val1,val2,...
                elif value.find(",") != -1:
                    list_str = value
                else:
                    # Something else entirely, return as string
                    return value
                # Recursesively parse the list elements
                return [
                    Descriptor._string_to_type(v.strip()) for v in list_str.split(",")
                ]

    @staticmethod
    def _parse_csv(line: str, separator: str = ",") -> dict:
        line = str(line)
        result = {}
        arg_tuples = line.split(separator)
        for arg in arg_tuples:
            if not arg:
                continue
            if "=" in arg:
                key, value = arg.split("=")
                result[key] = Descriptor._string_to_type(value)
            else:
                result[arg] = True
        return result

    @staticmethod
    def _remove_args_and_opts(line: str) -> str:
        line = str(line)
        if m := re.search(r"^([^[{]*)", line):
            line = m.group(1)
        return line

    @staticmethod
    def _parse_args_and_opts(line: str) -> tuple[str, dict, dict]:
        line = str(line)
        args = {}
        options = {}

        # Args: [arg1=val1,args2]
        if m := re.search(r"\[([^]]*)\]", line):
            args_str = m.group(1)
            args = Descriptor._parse_csv(args_str, ",")

        # Opts: {arg1=val1 args2}
        if m := re.search(r"\{([^}]+)\}", line):
            opts_str = m.group(1)
            options = Descriptor._parse_csv(opts_str, " ")

        # Cleanup the original string
        line = Descriptor._remove_args_and_opts(line)

        return [line, args, options]

    def __str__(self) -> str:
        """serialize basename + args + opts for transform consumption"""
        # Arguments have name and value, and are serialized as [name=value,...]
        args_str = (
            "[" + ",".join(f"{key}={value}" for key, value in self.args.items()) + "]"
            if self.args
            else ""
        )
        # Boolean options are serialized without the =True/False
        opts_str = (
            "{"
            + " ".join(
                f"{key}" if isinstance(value, bool) and value else f"{key}={value}"
                for key, value in self.opts.items()
            )
            + "}"
            if self.opts
            else ""
        )
        return f"{self.basename}{args_str}{opts_str}"


class PipelineDescriptor:
    """
    A descriptor for an optimization pipeline in YAML format.
    This class is responsible for parsing the pipeline description from a YAML file,
    and keeping a list of stages for consumption by the Driver.

    The format here is just text. The main job of this class is to handle includes,
    to verify that the files for the stages exist, normalize their paths, etc.
    The actual validation of the stages is left to the Driver and the stages themselves.

    Format is:
    Pipeline:
      - pass: PassName
      - transform: TransformFile.py[gen=generator_name,seq=sequence_name]{opt1=val1 opt2=val2}
      - transform: TransformFile.mlir
      - include: OtherPipeline.yaml
      ...
    """

    def __init__(self, desc: Descriptor):
        if not isinstance(desc, Descriptor):
            raise ValueError(
                f"PipelineDescriptor requires a Descriptor as input, got {type(desc)}"
            )
        self.descriptor = desc
        with open(desc.basename, "r") as f:
            self.pipeline_desc = yaml.safe_load(f)
        self.stages: list[str] = []
        self._parse_stages()
        if not self.stages:
            raise ValueError(
                f"Pipeline description file {desc.basename} does not contain a valid 'Pipeline'."
            )

    def _parse_stages(self) -> None:
        """
        Serialize the entire pipeline, including included pipelines, into a single list.
        """
        pipeline = self.pipeline_desc["Pipeline"]
        if not pipeline:
            raise ValueError(
                f"Pipeline description file {self.descriptor.basename} does not contain a 'Pipeline' key."
            )

        for stage in pipeline:
            key, value = next(iter(stage.items()))
            desc = Descriptor(value, type=key, base_path=self.descriptor.base_path)
            if desc.is_include():
                self._include_pipeline(desc)

            elif desc.is_transform():
                self.stages.append(desc)

            elif desc.is_pass():
                self.stages.append(desc)

            else:
                raise ValueError(f"Invalid stage in pipeline description: {desc}.")

    def _include_pipeline(self, desc: Descriptor) -> None:
        """
        Helper function to include another pipeline descriptor file.
        """
        included_pipeline = PipelineDescriptor(desc)
        self.stages.extend(included_pipeline.get_stages())

    def get_stages(self) -> list[str]:
        """Returns the list of stages in the pipeline."""
        return self.stages

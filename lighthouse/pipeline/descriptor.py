import yaml
import os

from lighthouse.pipeline.helper import remove_args_and_opts, update_filename


class PipelineDescriptor:
    """
    A descriptor for an optimization pipeline in YAML format.
    This class is responsible for parsing the pipeline description from a YAML file,
    and keeping a list of stages for comsumption by the Driver.

    The format here is just text. The main job of this class is to handle includes,
    to verify that the files for the stages exist, normalize their paths, etc.
    The actual validation of the stages is left to the Driver and the stages themselves.

    Format is:
    Pipeline:
      - pass: PassName
      - transform: TransformFile.py[gen=generator_name,seq=sequence_name]{opt1=val1 opt2=val2}
      - transform: TransformFile.mlir
      - include: OtherPipeline.yaml
      - bundle: BundleName
      ...
    """

    search_path = {
        ".py": "../schedule",
        ".yaml": "./descriptors",
    }

    def __init__(self, filename: str):
        self.filename = filename
        with open(filename, "r") as f:
            self.pipeline_desc = yaml.safe_load(f)
        self.stages: list[str] = []
        self._parse_stages()
        if not self.stages:
            raise ValueError(
                f"Pipeline description file {self.filename} does not contain a valid 'Pipeline'."
            )

    def _normalize_include_path(self, filename) -> str:
        """
        Finds the file in some standard locations, in order:
            * The path of the descriptor file that includes it. This allows for relative includes.
            * The path of the Lighthouse schedule module, where all the standard pipelines are located.
        """
        # First look in the same directory as the including file, to allow for relative includes.
        filename = remove_args_and_opts(filename)
        descriptor_path = os.path.normpath(os.path.dirname(self.filename))
        file = os.path.join(descriptor_path, filename)
        if os.path.exists(file):
            return file

        # If not found, look for an include path, based on the file extension.
        file_ext = os.path.splitext(file)[1]
        if file_ext not in self.search_path:
            raise ValueError(
                f"Included pipeline descriptor file does not exist: {filename} \
                    (searched in {descriptor_path})"
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
                (searched in {descriptor_path} and {schedule_module_path})"
        )

    def _parse_stages(self) -> None:
        """
        Serialize the entire pipeline, including included pipelines, into a single list.
        """
        pipeline = self.pipeline_desc["Pipeline"]
        if not pipeline:
            raise ValueError(
                f"Pipeline description file {self.filename} does not contain a 'Pipeline' key."
            )

        for stage in pipeline:
            if "include" in stage:
                # Includes recurr into the parser and return the stages.
                self._include_pipeline(stage["include"])

            elif "transform" in stage:
                # Transforms need to be MLIR files, and need to exist.
                filename = self._normalize_include_path(stage["transform"])
                if not filename.endswith(".mlir") and not filename.endswith(".py"):
                    raise ValueError(
                        f"Transform file must be an MLIR or Python file: {filename}"
                    )
                self.stages.append(update_filename(stage["transform"], filename))

            elif "pass" in stage:
                # Passes are just strings, let the pass manager validate.
                self.stages.append(stage["pass"])

            elif "bundle" in stage:
                # Bundle needs to exist in the driver, but to avoid cross import
                # we keep as text here. It's safe, as the stage will check if it exits.
                # TODO: Add a verification at this stage too.
                self.stages.append(stage["bundle"])

            else:
                raise ValueError(
                    f"Invalid stage in pipeline description: {stage}. Must be one of 'pass', 'transform', 'bundle' or 'include'."
                )

    def _include_pipeline(self, filename: str) -> None:
        """
        Helper function to include another pipeline descriptor file.
        """
        filename = self._normalize_include_path(filename)
        included_pipeline = PipelineDescriptor(filename)
        self.stages.extend(included_pipeline.get_stages())

    def get_stages(self) -> list[str]:
        """Returns the list of stages in the pipeline."""
        return self.stages

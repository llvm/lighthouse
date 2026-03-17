import yaml
import os


class PipelineDescriptor:
    """
    A descriptor for an optimization pipeline in YAML format.
    This class is responsible for parsing the pipeline description from a YAML file,
    and keeping a list of stages for comsumption by the Driver.

    Format is:
    Pipeline:
      - pass: PassName
      - transform: TransformFile.mlir
      - include: OtherPipeline.yaml
      - bundle: BundleName
      ...
    """

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
        """Uses the path of the includer to determine the path of the included."""
        return os.path.normpath(os.path.join(os.path.dirname(self.filename), filename))

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
                if not os.path.exists(filename):
                    raise ValueError(f"Transform file does not exist: {filename}")
                elif not filename.endswith(".mlir"):
                    raise ValueError(f"Transform file must be an MLIR file: {filename}")
                self.stages.append(filename)

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
        Include path is RELATIVE to the file including.
        """
        filename = self._normalize_include_path(filename)
        if not os.path.exists(filename):
            raise ValueError(
                f"Included pipeline descriptor file does not exist: {filename}"
            )
        included_pipeline = PipelineDescriptor(filename)
        self.stages.extend(included_pipeline.get_stages())

    def get_stages(self) -> list[str]:
        """Returns the list of stages in the pipeline."""
        return self.stages

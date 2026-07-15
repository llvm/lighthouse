import os

from pathlib import Path

from lighthouse.execution.target import TargetInfo


# Directory holding the pipeline descriptors packaged inside Lighthouse.
DESCRIPTORS_REPO = Path(__file__).parent / "descriptors"


def find_pipeline_file(
    target: TargetInfo,
    pipeline: str,
    dtype: str = "f32",
    base_path: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Find a pipeline descriptor file with the given name using target information to
    complete the following directory structure:
     - base_path/<arch>/<feature>/<pipeline>/<dtype>.yaml
     - base_path/<arch>/<pipeline>/<dtype>.yaml

    The lookup uses an externally provided ``base_path`` first, when given, to allow
    third-party repositories to ship their own descriptors. If no matching descriptor
    is found externally, the search falls back to Lighthouse's descriptors directory.

    Args:
        target: Target information to complete the directory structure.
        pipeline: The name of the pipeline descriptor file to find.
        dtype: The data type used to select the descriptor file.
        base_path: Optional external base directory searched before the
            Lighthouse descriptors directory.

    Returns:
        The full path to the pipeline descriptor file, or None if not found.
        The feature that was used to find the pipeline file, or None
        if no feature-specific file was found.
    """
    # If no name or target is provided, can't find the pipeline file.
    if not pipeline or not target:
        return None, None

    # Externally provided base paths take precedence, to allow third-party repos
    # to override or extend the packaged descriptors.
    if base_path is not None:
        file, feature = _find_pipeline_in_base(target, base_path, pipeline, dtype)
        if file is not None:
            return file, feature

    # Fall back to Lighthouse's own packaged descriptors directory.
    return _find_pipeline_in_base(target, DESCRIPTORS_REPO, pipeline, dtype)


def _find_pipeline_in_base(
    target: TargetInfo,
    base_path: str,
    pipeline: str,
    dtype: str,
) -> tuple[str | None, str | None]:
    """
    Search a single ``base_path`` for a pipeline descriptor file, following the
    directory structure documented in :func:`find_pipeline_file`.

    Returns (path, feature) if found, otherwise (None, None).
    """
    base_path = Path(base_path)

    # If the base path or arch directory doesn't exist, can't find the pipeline file.
    if not base_path.exists() or not (base_path / target.arch).exists():
        return None, None

    # Filter feature list based on available sub-directories in the base_dir
    arch_path = base_path / target.arch
    available_features = [
        name for name in os.listdir(arch_path) if (arch_path / name).is_dir()
    ]
    features = target.has_features(available_features)

    # First check if there's a pipeline file specific to the target features.
    for feature in features:
        pipeline_path = arch_path / feature / pipeline / f"{dtype}.yaml"
        if pipeline_path.exists():
            return str(pipeline_path), feature

    # Second, check if there's a pipeline file specific to the target architecture.
    if target.arch:
        pipeline_path = arch_path / pipeline / f"{dtype}.yaml"
        if pipeline_path.exists():
            return str(pipeline_path), None

    return None, None

import platform
import subprocess


class TargetInfo:
    """
    Struct to hold target architecture and feature information.
    Since this is used in a JIT context, we can safely assume the host
    architecture is the target architecture if not specified.

    Attributes:
        arch (str): The target architecture.
        features (list[str]): The list of CPU features (available on the target machine).
        filter (list[str]): The list of allowed features, if any (subset of `features`).
    """

    def __init__(
        self,
        arch: str | None = None,
        features: list[str] | None = None,
        filter: list[str] | None = None,
    ):
        self.arch = arch if arch is not None else platform.machine()
        self.features = features if features is not None else self._get_feature_list()
        # Pre-filter, if requested.
        if filter is not None:
            self.features = self.has_features(filter)

    def _get_feature_list(self) -> list[str]:
        """Get features from lscpu program"""
        flags = subprocess.run(
            "lscpu | grep Flags",
            capture_output=True,
            text=True,
            shell=True,
        ).stdout
        if not flags.startswith("Flags:"):
            raise RuntimeError(
                "Could not get CPU features from lscpu. "
                "Make sure lscpu is installed and available in PATH."
            )
        features = flags.split()[1:]  # Remove the "Flags:" prefix
        return features

    def has_features(self, filter: list[str]) -> list[str]:
        """
        Return a list of features that exist on both target and filter.
        """
        compatible = []
        for ext in self.features:
            if ext in filter:
                compatible.append(ext)
        return compatible

    def is_supported(self, hw_extension: str) -> bool:
        """
        Return True if the target supports the given hardware extension
        e.g., AMX or AVX512.
        """
        hw_extension = hw_extension.lower()
        return any(feature.startswith(hw_extension) for feature in self.features)

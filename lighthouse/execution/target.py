import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterInfo:
    width_bits: int
    count: int


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

    _cached_host: "TargetInfo | None" = None
    _override_features_stack: list[list[str] | None] = []
    _override_arch_stack: list[str | None] = []

    def __init__(
        self,
        arch: str | None = None,
        features: list[str] | None = None,
        filter: list[str] | None = None,
    ):
        if arch is None and self.__class__._override_arch_stack:
            arch = self.__class__._override_arch_stack[-1]
        if features is None and self.__class__._override_features_stack:
            override = self.__class__._override_features_stack[-1]
            if override is not None:
                features = list(override)

        self.arch = arch if arch is not None else platform.machine()
        self.features = features if features is not None else self._get_feature_list()
        # Pre-filter, if requested.
        if filter is not None:
            self.features = self.has_features(filter)

    @classmethod
    def host(cls) -> "TargetInfo":
        """Return a cached host TargetInfo honoring active test overrides."""
        if cls._override_features_stack or cls._override_arch_stack:
            return cls()
        if cls._cached_host is None:
            cls._cached_host = cls()
        return cls._cached_host

    @classmethod
    def reset_host_cache(cls) -> None:
        """Clear cached host target information."""
        cls._cached_host = None

    @classmethod
    @contextmanager
    def override(
        cls,
        *,
        features: list[str] | None = None,
        arch: str | None = None,
    ):
        """Temporarily override auto-detected host target info for tests."""
        cls._override_features_stack.append(
            None if features is None else list(features)
        )
        cls._override_arch_stack.append(arch)
        cls.reset_host_cache()
        try:
            yield
        finally:
            cls._override_features_stack.pop()
            cls._override_arch_stack.pop()
            cls.reset_host_cache()

    def _get_feature_list(self) -> list[str]:
        """Get CPU features from the host system."""
        if platform.system() == "Darwin":
            return self._get_feature_list_darwin()
        return self._get_feature_list_linux()

    @staticmethod
    def _get_feature_list_linux() -> list[str]:
        """Get features from lscpu (Linux)."""
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
        return flags.split()[1:]

    @staticmethod
    def _get_feature_list_darwin() -> list[str]:
        """Get features from sysctl (macOS)."""
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.cpu_features"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()

        # Apple Silicon: enumerate hw.optional.arm.* and hw.optional.armv8_* keys.
        result = subprocess.run(
            ["sysctl", "-a"],
            capture_output=True,
            text=True,
        )
        features = []
        for line in result.stdout.splitlines():
            if not line.startswith("hw.optional."):
                continue
            key, _, value = line.partition(":")
            if value.strip() not in ("1",):
                continue
            # Strip the "hw.optional." prefix.
            feat = key.split(".", 2)[-1]
            features.append(feat)
        if not features:
            raise RuntimeError(
                "Could not get CPU features from sysctl. "
                "Make sure sysctl is installed and available in PATH."
            )
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

    def vector_register_info(self) -> RegisterInfo | None:
        """Infer SIMD register info from target features."""
        if "avx512f" in self.features:
            return RegisterInfo(width_bits=512, count=32)
        if "avx2" in self.features or "avx" in self.features:
            return RegisterInfo(width_bits=256, count=16)
        if any(feature.startswith("sse") for feature in self.features):
            return RegisterInfo(width_bits=128, count=16)
        return None

    @property
    def vector_register_width_bits(self) -> int | None:
        info = self.vector_register_info()
        if info is None:
            return None
        return info.width_bits

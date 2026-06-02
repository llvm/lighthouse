"""
Central runtime configuration.

Settings are initialised from environment variables and can be overridden at
any time by assigning to the corresponding attribute on the `config` singleton.
"""

import os
from dataclasses import dataclass, field


def _bool_env(name: str, default: bool) -> bool:
    """Read an env var as a boolean flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes")


@dataclass
class _Config:
    """Mutable container for lighthouse runtime settings."""

    # Dump MLIR object file.
    mlir_dump_obj: bool = field(
        default_factory=lambda: _bool_env("LH_MLIR_DUMP_OBJ", False)
    )


# Module-level singleton – import and modify this directly.
config = _Config()

#!/usr/bin/env python3
"""Pre-commit hook enforcing kebab-case (hyphenated) YAML filenames.

YAML files (``*.yaml`` / ``*.yml``) must use lowercase, hyphen-separated
names, e.g. ``pack-and-tile.yaml`` rather than ``pack_and_tile.yaml`` or
``PackAndTile.yaml``. Filenames are passed on the command line by pre-commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

YAML_SUFFIXES = (".yaml", ".yml")

# A kebab-case stem: lowercase alphanumeric words joined by single hyphens.
KEBAB_CASE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def is_kebab_case(filename: str) -> bool:
    """Return True if the filename's stem is kebab-case."""
    stem = Path(filename).stem
    # Allow a single leading dot for hidden config files
    # (e.g. ".pre-commit-config.yaml").
    if stem.startswith("."):
        stem = stem[1:]
    return bool(KEBAB_CASE.match(stem))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*", help="YAML files to check.")
    args = parser.parse_args(argv)

    violations = [
        filename
        for filename in args.filenames
        if Path(filename).suffix in YAML_SUFFIXES and not is_kebab_case(filename)
    ]

    for filename in violations:
        print(
            f"{filename}: YAML filename '{Path(filename).name}' must be "
            "kebab-case (lowercase letters, digits, and hyphens only).",
            file=sys.stderr,
        )

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

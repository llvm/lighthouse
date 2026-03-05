#!/usr/bin/env bash

# Run this script from the root of the repository
# to run all pre-commit hooks and tests.
# This helps ensure that all checks are performed,
# and all files are formatted correctly before committing code.

set -ex

uv run pre-commit run --all-files
uv run lit -v .

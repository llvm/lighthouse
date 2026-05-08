#!/usr/bin/env bash

# Run this script from the root of the repository
# to run all pre-commit hooks and tests.
# This helps ensure that all checks are performed,
# and all files are formatted correctly before committing code.

set -ex

# Certain tests may require a larger stack size.
# Setting it here to a minimum expected to avoid cryptic
# runtime errors during test execution.
ulimit -s 16384

uv run pre-commit run --all-files
uv run lit -v .

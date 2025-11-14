"""
Pytest configuration for MLIR generation tests.

This file sets up fixtures and configuration for tests in this directory.
"""

import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_mlir_environment():
    """
    Set up MLIR environment variables for testing.
    This runs once per test session.
    """
    # Set environment variable for MLIR shared libraries if needed
    # The default empty string is fine for most cases
    if "LIGHTHOUSE_SHARED_LIBS" not in os.environ:
        os.environ["LIGHTHOUSE_SHARED_LIBS"] = ""

    yield

    # todo: cleanup
    pass

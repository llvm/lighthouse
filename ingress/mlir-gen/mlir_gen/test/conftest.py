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

    # Cleanup after all tests (if needed)
    pass


@pytest.fixture
def mlir_context():
    """
    Provide a fresh MLIR context for each test.
    """
    from mlir import ir

    return ir.Context()


@pytest.fixture
def sample_shapes():
    """
    Provide common tensor shapes for testing.
    """
    return [
        (4, 16),
        (8, 8),
        (16, 32),
        (1, 64),
    ]


@pytest.fixture
def sample_types():
    """
    Provide common element types for testing.
    """
    return ["f32", "f64"]

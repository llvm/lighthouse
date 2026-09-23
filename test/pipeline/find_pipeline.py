# RUN: %PYTHON %s | FileCheck %s

"""Discovery-path tests for descriptor repositories.

The tests are fully self-contained: both the "internal" packaged descriptors and
the "external" ``base_path`` are mocked with the persistent test fixtures.
"""

import os
from unittest import mock

from lighthouse.execution.target import TargetInfo
from lighthouse import pipeline

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
INTERNAL = os.path.join(FIXTURES, "internal")
EXTERNAL = os.path.join(FIXTURES, "external")


def normalize(path, external_root=None):
    """Map an absolute descriptor path."""
    if path is None:
        return "None"
    if external_root is not None and path.startswith(str(external_root)):
        return "EXTERNAL/" + os.path.relpath(path, external_root)
    if path.startswith(INTERNAL):
        return "INTERNAL/" + os.path.relpath(path, INTERNAL)
    return path


def show(label, result, external_root=None):
    path, feature = result
    print(f"{label}: {normalize(path, external_root)} feature={feature}")


# Temporarily point the lookup at our fixture repository instead of
# the real descriptors shipped with Lighthouse.
with mock.patch.object(pipeline.finder, "DESCRIPTORS_REPO", INTERNAL):
    # Feature-specific lookup inside the packaged descriptors.
    show(
        "feature-internal",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=["amx_bf16"]), "matmul", "bf16"
        ),
    )
    # CHECK: feature-internal: INTERNAL/x86_64/amx_bf16/matmul/bf16.yaml feature=amx_bf16

    # Architecture-level fallback inside the packaged descriptors.
    show(
        "arch-internal",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]), "matmul", "f32"
        ),
    )
    # CHECK: arch-internal: INTERNAL/x86_64/matmul/f32.yaml feature=None

    # An external base_path takes precedence over the packaged descriptors, even
    #    though the packaged descriptors also provide x86_64/matmul/f32.yaml.
    show(
        "external-precedence",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]), "matmul", "f32", base_path=EXTERNAL
        ),
        external_root=EXTERNAL,
    )
    # CHECK: external-precedence: EXTERNAL/x86_64/matmul/f32.yaml feature=None

    # An external base_path with no matching descriptor falls back to the packaged
    #    descriptors (the fixtures provide x86_64 but no "matvec" pipeline).
    show(
        "external-miss-fallback",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]), "matvec", "f32", base_path=EXTERNAL
        ),
        external_root=EXTERNAL,
    )
    # CHECK: external-miss-fallback: INTERNAL/x86_64/matvec/f32.yaml feature=None

    # A non-existent external base_path falls back gracefully (no exception).
    show(
        "external-missing-dir",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]),
            "matmul",
            "f32",
            base_path="/lh/does/not/exist",
        ),
    )
    # CHECK: external-missing-dir: INTERNAL/x86_64/matmul/f32.yaml feature=None

    # A pipeline file found directly under the architecture directory (no dtype subdir).
    show(
        "arch-file-internal",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]), "mat-vec", "f32"
        ),
    )
    # CHECK: arch-file-internal: INTERNAL/x86_64/mat-vec.yaml feature=None

    # An unknown pipeline name falls back to default.yaml when present for the architecture.
    show(
        "default-fallback",
        pipeline.find_pipeline_file(
            TargetInfo(arch="x86_64", features=[]), "no_such_pipeline", "f32"
        ),
    )
    # CHECK: default-fallback: INTERNAL/x86_64/default.yaml feature=None

    # An unknown architecture with no descriptors or default.
    show(
        "arch-not-found",
        pipeline.find_pipeline_file(
            TargetInfo(arch="unknown_arch", features=[]), "matmul", "f32"
        ),
    )
    # CHECK: arch-not-found: None feature=None

    # Missing pipeline name or target.
    show(
        "no-pipeline",
        pipeline.find_pipeline_file(TargetInfo(arch="x86_64", features=[]), "", "f32"),
    )
    # CHECK: no-pipeline: None feature=None

    show("no-target", pipeline.find_pipeline_file(None, "matmul", "f32"))
    # CHECK: no-target: None feature=None

# Test against the packaged Lighthouse descriptors repo (without mock).
# Since specialized pipelines for x86_64 were consolidated, any pipeline name falls back to default.yaml.
real_path, real_feature = pipeline.find_pipeline_file(
    TargetInfo(arch="x86_64", features=[]), "matmul", "f32"
)
print(
    f"packaged-x86_64-default: {os.path.basename(real_path) if real_path else None} feature={real_feature}"
)
# CHECK: packaged-x86_64-default: default.yaml feature=None

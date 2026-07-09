"""Shared XeGPU payload -> schedule dispatch.

This module centralizes the mechanism that decides *which* lowering schedule a
payload requires and builds it, so callers do not have to hard-code a specific
schedule (e.g. blindly assume a matmul). It is the common piece shared by the
KernelBench driver (``examples/xegpu/kernel_bench.py``) and the ``xeas``
assembler (``lighthouse.tools.xeas``).

The classification of a payload (matmul-like vs. elementwise-like) happens at
different IR levels for different callers, so this module only owns the parts
that are level-independent:

- The schedule-kind constants (:data:`MLP_SCHEDULE`, :data:`ELEMWISE_SCHEDULE`).
- :func:`select_schedule_kind`, which maps payload layer metadata (as produced
  by :func:`lighthouse.utils.mlir.inspect_payload`) to a schedule kind.
- :func:`build_payload_schedule`, which turns a schedule kind plus parameters
  into the matching transform schedule.
"""

from mlir import ir

from .mlp_schedule import mlp_schedule
from .elemwise_schedule import elemwise_schedule

#: Schedule kind for matmul / MLP-like payloads.
MLP_SCHEDULE = "mlp"
#: Schedule kind for elementwise-like payloads.
ELEMWISE_SCHEDULE = "elemwise"


def select_schedule_kind(func_metadata: dict) -> str:
    """Determine the schedule kind from payload layer metadata.

    Args:
        func_metadata: Metadata for a single payload function, as returned by
            :func:`lighthouse.utils.mlir.inspect_payload`. Only the ``"layers"``
            entry is inspected.

    Returns:
        :data:`MLP_SCHEDULE` if the payload contains at least one matmul layer,
        otherwise :data:`ELEMWISE_SCHEDULE` if it contains elementwise layers.

    Raises:
        ValueError: If the payload contains neither matmul nor elementwise
            layers.
    """
    layers = func_metadata["layers"]
    if layers.get("matmul"):
        return MLP_SCHEDULE
    if layers.get("elemwise"):
        return ELEMWISE_SCHEDULE
    raise ValueError(
        "Unsupported payload: expected at least one matmul or elementwise "
        f"layer, found layers: {dict(layers)}"
    )


def build_payload_schedule(
    schedule_kind: str,
    params: list[dict],
    payload_func_name: str = "payload",
    *,
    stop_at_stage: str | None = "",
    start_at_stage: str | None = "",
) -> ir.Module:
    """Build the transform schedule matching a payload's schedule kind.

    Args:
        schedule_kind: One of :data:`MLP_SCHEDULE` or :data:`ELEMWISE_SCHEDULE`.
        params: Per-layer schedule parameter dicts.
        payload_func_name: Name of the payload function to transform.
        stop_at_stage: Optional stage at which to stop the schedule.
        start_at_stage: Optional stage at which to resume the schedule
            (currently only ``"outlined"`` is supported, by both the MLP and the
            elementwise schedules).

    Returns:
        The transform schedule module.

    Raises:
        ValueError: If ``schedule_kind`` is not recognized.
    """
    stop_at_stage = stop_at_stage or ""
    start_at_stage = start_at_stage or ""
    if schedule_kind == MLP_SCHEDULE:
        return mlp_schedule(
            params=params,
            payload_func_name=payload_func_name,
            stop_at_stage=stop_at_stage,
            start_at_stage=start_at_stage,
        )
    if schedule_kind == ELEMWISE_SCHEDULE:
        return elemwise_schedule(
            params=params,
            payload_func_name=payload_func_name,
            stop_at_stage=stop_at_stage,
            start_at_stage=start_at_stage,
        )
    raise ValueError(f"Unsupported schedule kind: {schedule_kind!r}")

"""
Abstract base class for workloads.

Defines the expected interface for generic workload execution methods.
"""

from mlir import ir
from abc import ABC, abstractmethod
from typing import Optional


class Workload(ABC):
    """
    Abstract base class for workloads.

    A workload is defined by a fixed payload function and problem size.
    Different realizations of the workload can be obtained by altering the
    lowering schedule parameters.

    The MLIR payload function should take input arrays as memrefs and return
    nothing.
    """

    payload_function_name: str = "payload"
    benchmark_function_name: str = "bench_payload"

    @abstractmethod
    def shared_libs(self) -> list[str]:
        """Return a list of shared libraries required byt the execution engine."""
        pass

    @abstractmethod
    def payload_module(self) -> ir.Module:
        """Generate the MLIR module containing the payload function."""
        pass

    @abstractmethod
    def schedule_modules(
        self,
        stop_at_stage: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> list[ir.Module]:
        """
        Generate one or more MLIR modules containing the transform schedules.

        The `stop_at_stage` argument can be used to interrupt lowering at
        a desired IR level for debugging purposes.
        """
        pass

    @abstractmethod
    def get_complexity(self) -> tuple[int, int, int]:
        """
        Return the computational complexity of the workload.

        Returns a tuple (flop_count, memory_reads, memory_writes). Memory
        reads/writes are in bytes.
        """
        pass

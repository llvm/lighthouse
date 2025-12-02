"""
Abstract base class for workloads.

Defines the expected interface for generic workload execution methods.
"""

from mlir import ir
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional


class Workload(ABC):
    """
    Abstract base class for workloads.

    A workload is defined by a fixed payload function and problem size.
    Different realizations of the workload can be obtained by altering the
    lowering schedule.

    The MLIR payload function should take input arrays as memrefs and return
    nothing.
    """

    payload_function_name: str = "payload"

    @abstractmethod
    def requirements(self) -> list[str]:
        """Return a list of requirements for the execution engine."""
        pass

    @abstractmethod
    def payload_module(self) -> ir.Module:
        """Generate the MLIR module containing the payload function."""
        pass

    @abstractmethod
    def schedule_module(
        self,
        dump_kernel: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> ir.Module:
        """Generate the MLIR module containing the transform schedule."""
        pass

    @abstractmethod
    def get_input_arrays(self, execution_engine) -> list:
        """
        Return the input arrays for the payload function as memrefs.

        Allocation and initialization of the input arrays should be done here.
        """
        pass

    @contextmanager
    def allocate(self, execution_engine):
        """
        Allocate any necessary memory for the workload.

        Override this method if the workload requires memory management."""
        try:
            yield None
        finally:
            pass

    @abstractmethod
    def check_correctness(self, execution_engine, verbose: int = 0) -> bool:
        """Verify the correctness of the computation."""
        pass

    @abstractmethod
    def get_complexity(self) -> list:
        """
        Return the computational complexity of the workload.

        Returns a tuple (flop_count, memory_reads, memory_writes). Memory
        reads/writes are in bytes.
        """
        pass

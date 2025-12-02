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
    @contextmanager
    def allocate_inputs(self, execution_engine):
        """
        Context manager that allocates and returns payload input buffers.

        Returns the payload input buffers as memrefs that can be directly
        passed to the compiled payload function.

        On exit, frees any manually allocated memory (if any).
        """
        try:
            # Yield payload function input memrefs here.
            yield None
        finally:
            # Manually deallocate memory here (if needed).
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

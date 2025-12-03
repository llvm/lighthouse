"""
Abstract base class for workloads.

Defines the expected interface for generic workload execution methods.
"""

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from abc import ABC, abstractmethod
from contextlib import contextmanager
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

    @abstractmethod
    def shared_libs(self) -> list[str]:
        """Return a list of shared libraries required byt the execution engine."""
        pass

    @abstractmethod
    def payload_module(self) -> ir.Module:
        """Generate the MLIR module containing the payload function."""
        pass

    @abstractmethod
    def schedule_module(
        self,
        stop_at_stage: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> ir.Module:
        """
        Generate the MLIR module containing the transform schedule.

        The `stop_at_stage` argument can be used to interrupt lowering at
        a desired IR level for debugging purposes.
        """
        pass

    def lower_payload(
        self,
        dump_payload: Optional[str] = None,
        dump_schedule: bool = False,
        schedule_parameters: Optional[dict] = None,
    ) -> ir.Module:
        """
        Apply transform schedule to the payload module.

        Optionally dumps the payload IR at the desired level and/or the
        transform schedule to stdout.

        Returns the lowered payload module.
        """
        payload_module = self.payload_module()
        schedule_module = self.schedule_module(
            stop_at_stage=dump_payload, parameters=schedule_parameters
        )
        if not dump_payload or dump_payload != "initial":
            # apply schedule on payload module
            named_seq = schedule_module.body.operations[0]
            named_seq.apply(payload_module)
        if dump_payload:
            print(payload_module)
        if dump_schedule:
            print(schedule_module)
        return payload_module

    @abstractmethod
    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
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
    def check_correctness(
        self, execution_engine: ExecutionEngine, verbose: int = 0
    ) -> bool:
        """Verify the correctness of the computation."""
        pass

    @abstractmethod
    def get_complexity(self) -> tuple[int, int, int]:
        """
        Return the computational complexity of the workload.

        Returns a tuple (flop_count, memory_reads, memory_writes). Memory
        reads/writes are in bytes.
        """
        pass

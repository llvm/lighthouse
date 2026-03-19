"""
Abstract base class for workloads.

Defines the expected interface for generic workload execution methods.
"""

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional

from lighthouse.pipeline.opt import Driver, Stage, Transform, TransformStage


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
    def pipeline(
        self,
        stop_at_stage: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> list[str | Stage | ir.Module]:
        """
        Return a list of pipeline stages for lowering the payload.

        Each element can be a ready Stage object, an ir.Module (transform
        schedule), a Stage, or a string (pass name, bundle name, or file path).

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
        Apply the pipeline stages to the payload module using the Driver.

        Optionally dumps the payload IR at the desired level and/or the
        pipeline stages to stdout.

        Returns the lowered payload module.
        """
        payload_module = self.payload_module()
        stages = self.pipeline(
            stop_at_stage=dump_payload, parameters=schedule_parameters
        )
        if not isinstance(stages, list):
            raise TypeError(
                f"pipeline() must return a list, got {type(stages).__name__}"
            )
        if dump_payload and dump_payload == "initial":
            print(payload_module)
            return payload_module
        stages = [
            TransformStage(Transform(s), s.context) if isinstance(s, ir.Module) else s
            for s in stages
        ]
        driver = Driver(payload_module, stages)
        if dump_schedule:
            for stage in driver.pipeline:
                print(stage)
        module = driver.run()
        if dump_payload:
            print(module)
        return module

    @abstractmethod
    @contextmanager
    def allocate_inputs(self, execution_engine: ExecutionEngine):
        """
        Context manager that allocates and returns payload input buffers.

        Returns the payload input buffers as memrefs that can be directly
        passed to the compiled payload function.

        On exit, frees any manually allocated memory (if any).
        """
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

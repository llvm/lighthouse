from mlir import ir
from mlir.passmanager import PassManager


class Pipeline:
    """Defines a compilation pipeline that can be reused across workloads and ingresses.

    For example, this pipeline can be used as the default pipeline for all CPU workloads that do not specify a custom pipeline.
    The pipeline can be extended with additional passes as needed, and can be customized for different target architectures or workload characteristics.
    It can also be invoked in between transform schedules to canonicalize the IR and enable more optimization opportunities.
    """

    def __init__(self, context: ir.Context):
        self.context = context
        self.pm = PassManager("builtin.module", self.context)

    def add_bufferization(self) -> None:
        self.pm.add(
            "one-shot-bufferize{function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries}"
        )
        self.pm.add("drop-equivalent-buffer-results")
        self.pm.add("buffer-deallocation-pipeline")

    def add_llvm_lowering(self) -> None:
        self.pm.add("convert-linalg-to-loops")
        self.pm.add("convert-scf-to-cf")
        self.pm.add("convert-to-llvm")
        self.pm.add("reconcile-unrealized-casts")

    def add_cleanup(self) -> None:
        self.pm.add("cse")
        self.pm.add("canonicalize")

    def add_passes(self, passes: list[str]) -> None:
        for p in passes:
            self.pm.add(p)

    def run(self, module: ir.Module) -> ir.Module:
        assert module.context is self.context, (
            "Module context does not match pipeline context."
        )
        # IR is transformed in-place.
        self.pm.run(module.operation)
        return module

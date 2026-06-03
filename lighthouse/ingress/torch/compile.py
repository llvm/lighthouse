from collections.abc import Callable
from collections.abc import Sequence
import contextlib
from dataclasses import dataclass
from enum import Enum

from lighthouse import utils as lh_utils
from lighthouse.ingress.torch import import_from_model
from lighthouse.execution.runner import Runner
from lighthouse.execution import ExternalMemoryManager
from mlir import ir
from mlir.dialects import bufferization
from mlir.dialects import func
import torch


class TargetDialect(Enum):
    """Target MLIR dialect for importing PyTorch models."""

    # `Linalg` dialect ops using tensor semantics.
    LINALG_ON_TENSORS = "linalg-on-tensors"

    # `TOSA` dialect ops.
    TOSA = "tosa"

    # `StableHLO` dialect ops.
    STABLEHLO = "stablehlo"


@dataclass
class BufferMetadata:
    """Data buffer metadata."""

    shape: list[int]
    dtype: torch.dtype
    device: torch.device


@dataclass
class TorchMemoryManager(ExternalMemoryManager):
    """Pass-through memory manager for PyTorch tensors."""

    @contextlib.contextmanager
    def get_memrefs(self, inputs):
        """Convert PyTorch tensors to memref descriptors."""
        yield [lh_utils.torch.to_memref(t) for t in inputs]


class JITFunction:
    """
    Wrapper around JIT-compiled MLIR function.

    Manages lifetime of the compiled code and allocates result buffers.

    Args:
        module: MLIR module containing LLVM IR ops.
        results: Metadata of expected output buffers.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        entry_func: Name of the entry function.
        n_outputs: Number of last N outputs to return.
            Used to skip extra torch-mlir prepended results that might not
            be necessary.
        dump_obj_file: Target output object file.
    """

    def __init__(
        self,
        module: ir.Module,
        results: list[BufferMetadata],
        shared_libs: Sequence[str] = [],
        entry_func: str = "main",
        n_outputs: int | None = None,
        dump_obj_file: str = "",
    ):
        self.runner = Runner(
            module, mem_manager_cls=TorchMemoryManager, shared_libs=shared_libs
        )
        if dump_obj_file:
            self.runner.dump_object_file(dump_obj_file)
        self.entry_func = entry_func
        self.results = results
        self.n_outputs = n_outputs if n_outputs is not None else len(results)

    def __call__(
        self,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Allocate result buffers and call the jitted function.

        Args:
            args: Input tensors.

        Returns:
            Result tensors.
        """
        # Allocate empty buffers to store results.
        outs = [
            torch.empty(res.shape, dtype=res.dtype, device=res.device)
            for res in self.results
        ]

        # Prepare arguments according to MLIR backend's calling convention:
        # input data followed by output storage buffers.
        all_tensors = [arg.detach() for arg in args]
        all_tensors.extend(outs)
        self.runner.execute(self.entry_func, all_tensors)

        # Return only the outputs corresponding to the FX graph's actual return
        # values. torch-mlir may prepend extra results for in-place state
        # mutations (e.g. BatchNorm running statistics), which Dynamo does not
        # expect in the backend callable's return value.
        return outs[len(self.results) - self.n_outputs :]


class MLIRBackend:
    """
    A PyTorch backend via MLIR.

    The backend exports a PyTorch model into an MLIR module.
    The entry function is first preprocessed to simplify data management. Notably:
      - MLIR entry function is marked private to enable more optimizations
      - MLIR result buffers are moved into arguments
    Use of output argument prevents need for cross Python-MLIR buffer lifetime
    management. The output buffer are allocated outside of MLIR (see 'JITFunction')
    on the target device.

    Then the provided compilation function partially lowers the imported program
    to LLVM IR dialect before the final executable is created.

    Args:
        device: Target device.
        fn_compile: Function to lower imported MLIR to LLVM IR dialect.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
            If not provided, a new default context is created.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        entry_func: Name of the entry function.
        dump_obj_file: Target output object file.
    """

    def __init__(
        self,
        device: torch.device,
        fn_compile: Callable[[ir.Module], ir.Module],
        dialect: TargetDialect = TargetDialect.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
        entry_func: str = "main",
        dump_obj_file: str = "",
    ):
        self.device = device
        self.fn_compile = fn_compile
        self.dialect = dialect
        self.ctx = ir_context if ir_context is not None else ir.Context()
        self.shared_libs = list(shared_libs)
        self.entry_func = entry_func
        self.dump_obj_file = dump_obj_file

    def get_entry_func(self, module: ir.Module) -> func.FuncOp | None:
        """
        Find entry function in MLIR module.

        Args:
            module: MLIR module.

        Returns:
            Entry function operation if present, None otherwise.
        """
        assert len(module.operation.regions) == 1, "Expected module with one region"
        assert len(module.operation.regions[0].blocks) == 1, (
            "Expected module with one block"
        )

        for op in module.operation.regions[0].blocks[0].operations:
            if (
                isinstance(op.opview, func.FuncOp)
                and op.opview.name.value == self.entry_func
            ):
                return op.opview
        return None

    def get_results(self, func_op: func.FuncOp) -> list[BufferMetadata]:
        """
        Extract metadata about function operation results.

        Args:
            func_op: MLIR function op.

        Returns:
            A list of metadata per result.
        """
        results = []
        for res in func_op.type.results:
            assert isinstance(res, ir.RankedTensorType), "Expected ranked tensor output"
            res_dtype = lh_utils.torch.dtype_from_mlir_type(res.element_type)
            results.append(
                BufferMetadata(shape=res.shape, dtype=res_dtype, device=self.device)
            )
        return results

    def is_symbolic(self, tensor: torch.Tensor) -> bool:
        """Check if a tensor has any symbolic dimensions."""
        return isinstance(tensor, (torch.SymFloat, torch.SymInt, torch.SymBool))

    def get_mlir(
        self, model: torch.nn.Module, example_inputs: list[torch.Tensor]
    ) -> ir.Module:
        """
        Convert PyTorch model to MLIR IR.

        Args:
            model: PyTorch model.
            example_inputs: Inputs to the model.
        Returns:
            MLIR module.
        """
        # Suppress importer's messages.
        # 'torch_mlir' prints to STDOUT which can be noisy.
        with contextlib.redirect_stdout(None):
            mlir_mod = import_from_model(
                model,
                sample_args=example_inputs,
                dialect=self.dialect.value,
                ir_context=self.ctx,
            )
        return mlir_mod

    def move_results_to_args(self, func_op: func.FuncOp):
        """
        Move function results to arguments.
        The function is modified in-place.

        Args:
            func_op: Function op to modify.
        """
        results = func_op.type.results
        if len(results) == 0:
            return

        assert all(isinstance(res, ir.RankedTensorType) for res in results), (
            "Expected only ranked tensor results"
        )

        with func_op.context, func_op.location:
            # Current bufferization can't handle return op fed by new tensor values
            # created using 'materialize_in_destination' op (missing region branch
            # interface).
            #
            # Create equivalent memref buffers, instead.
            # PyTorch tensors are later converted to ranked memrefs anyway.
            memref_bufs = [
                ir.MemRefType.get(res.shape, res.element_type) for res in results
            ]

            # Append results to function args and its block args.
            new_func_type = ir.FunctionType.get(
                inputs=[*func_op.type.inputs, *memref_bufs], results=[]
            )
            func_op.function_type = ir.TypeAttr.get(new_func_type)
            for res in memref_bufs:
                func_op.entry_block.add_argument(res, func_op.location)

            # Ensure outputs are written to the new result arguments.
            return_op: func.ReturnOp = func_op.entry_block.operations[-1]
            with (
                ir.InsertionPoint.at_block_terminator(func_op.entry_block),
                return_op.location,
            ):
                for idx, arg in enumerate(func_op.arguments[-len(results) :]):
                    bufferization.materialize_in_destination(
                        None,
                        return_op.operands[idx],
                        arg,
                        restrict=True,
                        writable=True,
                    )
                func.return_([])
                return_op.erase()

    def preprocess_func(self, func_op: func.FuncOp):
        """
        Modify an MLIR function to align with Python-MLIR calling convention.
        The function is modified in-place.

        Args:
            func_op: Function to modify.
        """
        # Mark the function private as the symbol will not be exposed externally.
        # Private functions allow for more rewrites.
        func_op.sym_visibility = ir.StringAttr.get("private", func_op.context)
        # Add extra arguments to store results in external buffers.
        self.move_results_to_args(func_op)

    def compile_mlir(self, module: ir.Module) -> ir.Module:
        """
        Lower an MLIR module.

        Args:
            module: MLIR module to be compiled.
        Returns:
            Lowered MLIR module.
        """
        return self.fn_compile(module)

    def __call__(
        self, model: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[[list[torch.Tensor]], list[torch.Tensor]]:
        """
        Import a PyTorch model into MLIR and return a compiled function.

        Args:
            model: Traced PyTorch model.
            example_inputs: Example input tensors.

        Returns:
            Callable function.
        """
        if any(self.is_symbolic(in_tensor) for in_tensor in example_inputs):
            raise ValueError(
                "Dynamic shapes are not supported"
                " - consider using 'torch.compile(..., dynamic=False)'"
            )

        # Count the FX graph's actual outputs before importing to MLIR.
        # torch-mlir's import may prepend extra results for in-place state
        # mutations that the FX graph does not expose as outputs.
        output_node = next(n for n in model.graph.nodes if n.op == "output")
        n_fx_outputs = len(output_node.args[0])

        mlir_mod = self.get_mlir(model, example_inputs)

        func_op = self.get_entry_func(mlir_mod)
        if func_op is None:
            raise ValueError(f"Failed to find MLIR entry: {self.entry_func}")
        # Metadata about function returns is stored for later output buffer allocation.
        results = self.get_results(func_op)
        # Modify MLIR entry function to align with calling convention.
        self.preprocess_func(func_op)

        mlir_mod = self.compile_mlir(mlir_mod)

        return JITFunction(
            mlir_mod,
            results,
            shared_libs=self.shared_libs,
            entry_func=self.entry_func,
            n_outputs=n_fx_outputs,
            dump_obj_file=self.dump_obj_file,
        )


def cpu_backend(
    fn_compile: Callable[[ir.Module], ir.Module],
    dialect: TargetDialect = TargetDialect.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    entry_func: str = "main",
    dump_obj_file: str = "",
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable]:
    """
    CPU backend for JIT-compiling a PyTorch model using MLIR.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        entry_func: Name of the entry function.
        dump_obj_file: Target output object file.

    Returns:
        A torch.compile backend object.
    """
    return MLIRBackend(
        torch.device("cpu"),
        fn_compile,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
        entry_func=entry_func,
        dump_obj_file=dump_obj_file,
    )


def gpu_backend(
    fn_compile: Callable[[ir.Module], ir.Module],
    device: torch.device,
    dialect: TargetDialect = TargetDialect.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    entry_func: str = "main",
    dump_obj_file: str = "",
) -> Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable]:
    """
    GPU backend for JIT-compiling a PyTorch model using MLIR.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        device: Target GPU device.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        entry_func: Name of the entry function.
        dump_obj_file: Target output object file.

    Returns:
        A torch.compile backend object.
    """
    assert device.type in ("cuda", "rocm", "xpu"), "Expected a GPU device"

    return MLIRBackend(
        device,
        fn_compile,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
        entry_func=entry_func,
        dump_obj_file=dump_obj_file,
    )

from collections.abc import Callable, Iterable, Sequence
import functools
import inspect
from typing import Any

import torch
import torch.nn as nn
from torch_mlir.fx import OutputType

from mlir import ir
from mlir.execution_engine import ExecutionEngine
from lighthouse.ingress.torch import import_from_model
from lighthouse import utils as lh_utils


class JITModule:
    def __init__(
        self,
        fn_compile_mlir: Callable[[ir.Module, ir.Context], ir.Module],
        module: nn.Module,
        dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
    ):
        """
        Initialize the JITModule object.
        Typically called from the `@lighthouse.runtime.torch.jit` decorator.

        Args:
            fn_compile_mlir: Function to lower imported MLIR to LLVM IR dialect.
            module: PyTorch module to be compiled through MLIR.
            dialect: The target dialect for MLIR IR imported from PyTorch module.
            ir_context: An optional MLIR context to use for compilation.
                If not provided, a new default context is created.
            shared_libs: Paths to external runtime libraries used to execute
                compiled MLIR function.
        """
        self.fn_compile = fn_compile_mlir
        self.module = module
        self.dialect = OutputType.get(dialect)
        self.ctx = ir_context
        if self.ctx is None:
            self.ctx = ir.Context()
        self.shared_libs = shared_libs

    def get_module(self) -> nn.Module:
        """Return initialized PyTorch module."""
        return self.module

    def get_dialect(self) -> OutputType:
        """Return target MLIR dialect."""
        return self.dialect

    def get_context(self) -> ir.Context:
        """Return MLIR context used by the jitted module."""
        return self.ctx

    def get_shared_libs(self) -> Sequence[str]:
        """Return runtime shared libraries."""
        return self.shared_libs

    def __call__(
        self,
        *args: Sequence[torch.Tensor] | object,
        module_args: Iterable | None = None,
        **kwargs,
    ) -> Any:
        """
        Jit the PyTorch module and call the MLIR function.

        Args:
            args: The positional arguments to pass the MLIR function.
                If all arguments are PyTorch tensors, then they are converted
                to packed C-type arguments before passing to the MLIR function.
                Otherwise, `args` are passed directly as is.
            module_args: The optional positional arguments to the Pytorch module
                required to jit into MLIR.
                If not provided, `args` are used instead.
            kwargs: The keyword arguments to the PyTorch module required to jit
                into MLIR.

        Returns:
            Any: The result of the MLIR function call.
        """

        if module_args is None:
            module_args = args

        # TODO: Add caching.
        mlir_mod = import_from_model(
            self.module,
            sample_args=module_args,
            sample_kwargs=kwargs,
            dialect=self.dialect,
            ir_context=self.ctx,
        )
        mlir_mod = self.fn_compile(mlir_mod, self.ctx)

        eng = ExecutionEngine(mlir_mod, opt_level=3, shared_libs=self.shared_libs)
        eng.initialize()
        fn = eng.lookup("main")

        mlir_args = args
        if all(torch.is_tensor(arg) for arg in args):
            mlir_args = (lh_utils.torch.to_packed_args(args),)

        return fn(*mlir_args)


def jit(
    fn_compile: Callable[[ir.Module, ir.Context], ir.Module],
    module: type[nn.Module] | nn.Module | None = None,
    *,
    dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    **kwargs,
) -> JITModule | Callable:
    """
    Decorator for JIT-compiling a PyTorch module using MLIR.

    When a PyTorch module class is passed, the module must be initialized first.
    Only calls to instantiated module objects are JIT-compiled.

    When a jitted MLIR function is called, input arguments are implicitly converted
    into packed C-type arguments if all inputs are PyTorch tensors.
    Otherwise, the input arguments are directly passed to the jitted function.

    The jitted function signature depends on the provided MLIR compilation function
    `fn_compile` and may differ from the original PyTorch module call signature.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module and context, and returns
            an MLIR module with transformed IR.
        module: PyTorch module to be compiled with MLIR.
            If a class or None, a decorator is returned.
        dialect: The target dialect for MLIR IR imported from PyTorch module.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        kwargs: The keyword arguments for the PyTorch module constructor.

    Returns:
        object: A JITModule object or a partially bound decorator.
    """

    def class_decorator(
        cls_module: type[nn.Module],
        *args,
        dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
        **kwargs,
    ) -> JITModule:
        module = cls_module(*args, **kwargs)
        return JITModule(
            fn_compile,
            module,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
        )

    if module is None:
        # Return a partial decorator with bound compilation function.
        return functools.partial(
            jit,
            fn_compile,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
            **kwargs,
        )
    if inspect.isclass(module):
        # Return a class decorator which accepts further arguments to
        # first construct a PyTorch module object before creating
        # a JITModule object.
        return functools.partial(
            class_decorator,
            module,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
            **kwargs,
        )
    # Directly create a JITModule object from an instantiated PyTorch module.
    return JITModule(
        fn_compile,
        module,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
    )

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


class JITModel:
    def __init__(
        self,
        fn_compile_mlir: Callable[[ir.Module], ir.Module],
        model: nn.Module,
        dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
    ):
        """
        Initialize the JITModel object.
        Typically called from the `@lighthouse.runtime.torch.jit` decorator.

        Args:
            fn_compile_mlir: Function to lower imported MLIR to LLVM IR dialect.
            model: PyTorch model to be compiled through MLIR.
            dialect: The target dialect for MLIR IR imported from PyTorch model.
            ir_context: An optional MLIR context to use for compilation.
                If not provided, a new default context is created.
            shared_libs: Paths to external runtime libraries used to execute
                compiled MLIR function.
        """
        self.fn_compile = fn_compile_mlir
        self.model = model
        self.dialect = OutputType.get(dialect)
        self.ctx = ir_context if ir_context is not None else ir.Context()
        self.shared_libs = shared_libs

    def __call__(
        self,
        *args: Sequence[torch.Tensor] | object,
        model_args: Iterable | None = None,
        **kwargs,
    ) -> Any:
        """
        Jit the PyTorch model and call the MLIR function.

        Args:
            args: The positional arguments to pass the MLIR function.
                If all arguments are PyTorch tensors, then they are converted
                to packed C-type arguments before passing to the MLIR function.
                Otherwise, `args` are passed directly as is.
            model_args: The optional positional arguments to the Pytorch model
                required to jit into MLIR.
                If not provided, `args` are used instead.
            kwargs: The keyword arguments to the PyTorch model required to jit
                into MLIR.

        Returns:
            Any: The result of the MLIR function call.
        """

        if model_args is None:
            model_args = args

        # TODO: Add caching.
        mlir_mod = import_from_model(
            self.model,
            sample_args=model_args,
            sample_kwargs=kwargs,
            dialect=self.dialect,
            ir_context=self.ctx,
        )
        mlir_mod = self.fn_compile(mlir_mod)

        eng = ExecutionEngine(mlir_mod, opt_level=3, shared_libs=self.shared_libs)
        eng.initialize()
        fn = eng.lookup("main")

        mlir_args = args
        if all(torch.is_tensor(arg) for arg in args):
            mlir_args = (lh_utils.torch.to_packed_args(args),)

        return fn(*mlir_args)


def jit(
    fn_compile: Callable[[ir.Module], ir.Module],
    model: type[nn.Module] | nn.Module | None = None,
    *,
    dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
    ir_context: ir.Context | None = None,
    shared_libs: Sequence[str] = [],
    **kwargs,
) -> JITModel | Callable:
    """
    Decorator for JIT-compiling a PyTorch model using MLIR.

    When a PyTorch model class is passed, the model must be initialized first.
    Only calls to instantiated model objects are JIT-compiled.

    When a jitted MLIR function is called, input arguments are implicitly converted
    into packed C-type arguments if all inputs are PyTorch tensors.
    Otherwise, the input arguments are directly passed to the jitted function.

    The jitted function signature depends on the provided MLIR compilation function
    `fn_compile` and may differ from the original PyTorch model call signature.

    Args:
        fn_compile: Function to compile imported MLIR to LLVM IR dialect.
            The function accepts an MLIR module, and returns an MLIR module with
            transformed IR.
        model: PyTorch model to be compiled with MLIR.
            If a class or None, a decorator is returned.
        dialect: The target dialect for MLIR IR imported from PyTorch model.
        ir_context: An optional MLIR context to use for compilation.
        shared_libs: Paths to external runtime libraries used to execute
            compiled MLIR function.
        kwargs: The keyword arguments for the PyTorch model constructor.

    Returns:
        object: A JITModel object or a partially bound decorator.
    """

    def class_decorator(
        cls_model: type[nn.Module],
        *args,
        dialect: OutputType | str = OutputType.LINALG_ON_TENSORS,
        ir_context: ir.Context | None = None,
        shared_libs: Sequence[str] = [],
        **kwargs,
    ) -> JITModel:
        model = cls_model(*args, **kwargs)
        return JITModel(
            fn_compile,
            model,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
        )

    if model is None:
        # Return a partial decorator with bound compilation function.
        return functools.partial(
            jit,
            fn_compile,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
            **kwargs,
        )
    if inspect.isclass(model):
        # Return a class decorator which accepts further arguments to
        # first construct a PyTorch model object before creating
        # a JITModel object.
        return functools.partial(
            class_decorator,
            model,
            dialect=dialect,
            ir_context=ir_context,
            shared_libs=shared_libs,
            **kwargs,
        )
    # Directly create a JITModel object from an instantiated PyTorch model.
    return JITModel(
        fn_compile,
        model,
        dialect=dialect,
        ir_context=ir_context,
        shared_libs=shared_libs,
    )

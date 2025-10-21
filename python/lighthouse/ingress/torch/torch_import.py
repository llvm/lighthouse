import importlib
import importlib.util
from pathlib import Path
from typing import Iterable, Mapping

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is required to use the torch import functionality. "
        "Please run 'uv pip install .[torch-mlir]'"
    ) from e

try:
    from torch_mlir import fx
    from torch_mlir.fx import OutputType
except ImportError as e:
    raise ImportError(
        "torch-mlir is required to use the torch import functionality. "
        "Please run 'uv pip install .[torch-mlir]'"
    ) from e

from mlir import ir

def import_from_model(
    model: nn.Module,
    sample_args : Iterable,
    sample_kwargs : Mapping = None,
    dialect : OutputType | str = OutputType.LINALG_ON_TENSORS,
    ir_context : ir.Context | None = None,
    **kwargs,
) -> str | ir.Module:
    """Import a PyTorch nn.Module into MLIR.

    The function uses torch-mlir's FX importer to convert the given PyTorch model
    into an MLIR module in the specified dialect. The user has to provide sample
    input arguments (e.g. a torch.Tensor with the correct shape).

    Args:
        model (nn.Module): The PyTorch model to import.
        sample_args (Iterable): Sample input arguments to the model.
        sample_kwargs (Mapping, optional): Sample keyword arguments to the model.
        dialect (torch_mlir.fx.OutputType | {"linalg-on-tensors", "torch", "tosa"}):
            The target dialect for the imported MLIR module. Defaults to
            ``OutputType.LINALG_ON_TENSORS``.
        ir_context (ir.Context, optional): An optional MLIR context to use for parsing
            the module. If not provided, the module is returned as a string.
        **kwargs: Additional keyword arguments passed to the ``torch_mlir.fx.export_and_import`` function.
    
    Returns:
        str | ir.Module: The imported MLIR module as a string or an ir.Module if `ir_context` is provided.
    
    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from lighthouse.ingress.torch_import import import_from_model
        >>> class SimpleModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.Linear(10, 5)
        ...     def forward(self, x):
        ...         return self.fc(x)
        >>> model = SimpleModel()
        >>> sample_input = (torch.randn(1, 10),)
        >>> #
        >>> # option 1: get MLIR module as a string
        >>> mlir_module : str = import_from_model(model, sample_input, dialect="linalg-on-tensors")
        >>> print(mlir_module) # prints the MLIR module in linalg-on-tensors dialect
        >>> # option 2: get MLIR module as an ir.Module
        >>> ir_context = ir.Context()
        >>> mlir_module_ir : ir.Module = import_from_model(model, sample_input, dialect="tosa", ir_context=ir_context)
        >>> # ... run pm.Pipeline on the ir.Module ...
    """
    if dialect == "linalg":
        raise ValueError(
            "Dialect 'linalg' is not supported. Did you mean 'linalg-on-tensors'?"
        )

    if sample_kwargs is None:
        sample_kwargs = {}

    model.eval()
    module = fx.export_and_import(
        model, *sample_args, output_type=dialect, **sample_kwargs, **kwargs
    )

    text_module = str(module)
    if ir_context is None:
        return text_module
    # Cross boundary from torch-mlir's mlir to environment's mlir
    return ir.Module.parse(text_module, context=ir_context)


def import_from_file(
    filepath: str | Path,
    model_class_name: str = "Model",
    init_args_fn_name: str | None = "get_init_inputs",
    inputs_args_fn_name: str = "get_inputs",
    state_path : str | Path | None = None,
    dialect : OutputType | str = OutputType.LINALG_ON_TENSORS,
    ir_context : ir.Context | None = None,
    **kwargs,
) -> str | ir.Module:
    """Load a PyTorch nn.Module from a file and import it into MLIR.

    The function takes a `filepath` to a Python file containing the model definition,
    along with the names of functions to get model init arguments and sample inputs.
    The function imports the model class on its own, instantiates it, and passes
    it ``torch_mlir`` to get a MLIR module in the specified `dialect`.

    Args:
        filepath (str | Path): Path to the Python file containing the model definition.
        model_class_name (str, optional): The name of the model class in the file.
            Defaults to "Model".
        init_args_fn_name (str | None, optional): The name of the function in the file
            that returns the arguments for initializing the model. If None, the model
            is initialized without arguments. Defaults to "get_init_inputs".
        inputs_args_fn_name (str, optional): The name of the function in the file that
            returns the sample input arguments for the model. Defaults to "get_inputs".
        state_path (str | Path | None, optional): Optional path to a file containing
            the model's ``state_dict``. Defaults to None.
        dialect (torch_mlir.fx.OutputType | {"linalg-on-tensors", "torch", "tosa"}, optional):
            The target dialect for the imported MLIR module. Defaults to
            ``OutputType.LINALG_ON_TENSORS``.
        ir_context (ir.Context, optional): An optional MLIR context to use for parsing
            the module. If not provided, the module is returned as a string.
        **kwargs: Additional keyword arguments passed to the ``torch_mlir.fx.export_and_import`` function.
    
    Returns:
        str | ir.Module: The imported MLIR module as a string or an ir.Module if `ir_context` is provided.
    
    Examples:
        Given a file `path/to/model_file.py` with the following content:
        ```python
        import torch
        import torch.nn as nn

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            def forward(self, x):
                return self.fc(x)

        def get_inputs():
            return (torch.randn(1, 10),)
        ```

        The import script would look like:
        >>> from lighthouse.ingress.torch_import import import_from_file
        >>> # option 1: get MLIR module as a string
        >>> mlir_module : str = import_from_file(
        ...     "path/to/model_file.py",
        ...     model_class_name="MyModel",
        ...     init_args_fn_name=None,
        ...     dialect="linalg-on-tensors"
        ... )
        >>> print(mlir_module) # prints the MLIR module in linalg-on-tensors dialect
        >>> # option 2: get MLIR module as an ir.Module
        >>> ir_context = ir.Context()
        >>> mlir_module_ir : ir.Module = import_from_file(
        ...     "path/to/model_file.py",
        ...     model_class_name="MyModel",
        ...     init_args_fn_name=None,
        ...     dialect="linalg-on-tensors",
        ...     ir_context=ir_context
        ... )
        >>> # ... run pm.Pipeline on the ir.Module ...
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    module_name = filepath.stem

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = getattr(module, model_class_name, None)
    if model is None:
        raise ValueError(f"Model class '{model_class_name}' not found in {filepath}")

    if init_args_fn_name is None:
        init_args_fn = lambda *args, **kwargs: ()
    else:
        init_args_fn = getattr(module, init_args_fn_name, None)
        if init_args_fn is None:
            raise ValueError(f"Init args function '{init_args_fn_name}' not found in {filepath}")

    inputs_args_fn = getattr(module, inputs_args_fn_name, None)
    if inputs_args_fn is None:
        raise ValueError(f"Inputs args function '{inputs_args_fn_name}' not found in {filepath}")

    nn_model : nn.Module = model(*init_args_fn())
    if state_path is not None:
        state_dict = torch.load(state_path)
        nn_model.load_state_dict(state_dict)

    return import_from_model(nn_model, *inputs_args_fn(), dialect=dialect, ir_context=ir_context, **kwargs)

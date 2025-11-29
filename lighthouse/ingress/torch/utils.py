from types import ModuleType
from typing import Any


def load_and_run_callable(
    module: ModuleType,
    symbol_name: str,
    error_msg: str | None = None,
):
    """Helper to load and run a callable from a module by its symbol name.

    Args:
        module (ModuleType): The python module to load the callable from.
        symbol_name (str): The name of the callable symbol to load.
        error_msg (str | None): Custom error message to use when raising an error
            for missing symbol. If not provided, a default message will be used.

    Returns:
        Any: The result of calling the loaded callable.
    """
    func = getattr(module, symbol_name, None)
    if func is None:
        if error_msg:
            raise ValueError(error_msg)
        raise ValueError(
            f"Symbol '{symbol_name}' not found in module '{module.__name__}'"
        )
    if not callable(func):
        raise ValueError(f"Symbol '{symbol_name}' is not callable")
    return func()


def maybe_load_and_run_callable(
    module: ModuleType,
    symbol_name: str | None,
    default: Any,
    error_msg: str | None = None,
):
    """Helper to conditionally load and run a callable from a module by its symbol name.

    If `symbol_name` is None, the function returns the provided default value. Otherwise
    it calls ``load_and_run_callable`` with the provided arguments.
    """
    if symbol_name is None:
        return default
    return load_and_run_callable(module, symbol_name, error_msg=error_msg)

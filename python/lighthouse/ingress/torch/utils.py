from types import ModuleType
from typing import Any


def load_and_run_callable(
    module: ModuleType,
    symbol_name: str | None,
    raise_on_missing: bool = True,
    default: Any | None = None,
    error_msg: str | None = None,
):
    """Helper to load and run a callable from a module by its symbol name.

    Args:
        module (ModuleType): The python module to load the callable from.
        symbol_name (str | None): The name of the callable symbol to load.
            If not specified (None), the function will either return the `default` value
            or raise an error based on `raise_on_missing`.
        raise_on_missing (bool): Whether to raise an error if the symbol is missing
            or not specified. Default is True.
        default (Any | None): The default value to return if the symbol is missing
            and `raise_on_missing` is False.
        error_msg (str | None): Custom error message to use when raising an error
            for missing symbol. If not provided, a default message will be used.

    Returns:
        Any: The result of calling the loaded callable or `default` if the symbol is
          missing and `raise_on_missing` is False.
    """
    if symbol_name is None:
        if raise_on_missing:
            if error_msg:
                raise ValueError(error_msg)
            raise ValueError("No symbol name provided")
        return default
    func = getattr(module, symbol_name, None)
    if func is None:
        if raise_on_missing:
            if error_msg:
                raise ValueError(error_msg)
            raise ValueError(
                f"Symbol '{symbol_name}' not found in module '{module.__name__}'"
            )
        return default
    if not callable(func):
        raise ValueError(f"Symbol '{symbol_name}' is not callable")
    return func()

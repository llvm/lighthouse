from typing import TypeVar, Generic, Callable

from collections.abc import Mapping

K = TypeVar("K")
V = TypeVar("V")
W = TypeVar("W")


class LazyChainMap(Mapping, Generic[K, V, W]):
    """A mapping that applies a function to the values of an underlying dictionary on access."""

    def __init__(self, data: dict[K, V], func: Callable[[V], W]):
        self._data = data
        self._func = func

    def __getitem__(self, key):
        # Access the underlying data and apply the transformation
        value = self._data[key]
        return self._func(value)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def string_to_type(value: any) -> str | int | float | bool | list | tuple:
    """
    Convert a string value to its appropriate type (int, float, bool, list, tuple, or str).
    If the argument isn't a string, convert to string first.
    If the argument cannot be converted to string, return None.
    None and empty strings are returned as-is.
    """
    if value is None:
        return None
    try:
        value = str(value)
    except Exception:
        return None
    if not value:
        return value
    if value.lower() == "none":
        return None

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    # List of values, e.g. [val1,val2,...]
    if value.startswith("[") and value.endswith("]"):
        list_str = value[1:-1]
        if not list_str:
            return []
        # Recursively parse the list elements
        return [string_to_type(v.strip()) for v in list_str.split(",")]

    # Tuple of values, e.g. (val1,val2,...)
    if value.startswith("(") and value.endswith(")"):
        tuple_str = value[1:-1]
        if not tuple_str:
            return ()
        # Recursively parse the tuple elements
        return tuple(string_to_type(v.strip()) for v in tuple_str.split(","))

    # Something else entirely, return as string
    return value

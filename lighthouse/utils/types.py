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

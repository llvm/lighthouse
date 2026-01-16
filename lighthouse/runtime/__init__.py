__all__ = ["torch"]


import sys
import importlib


def __getattr__(name):
    """Enable lazy loading of submodules."""

    if name in __all__:
        # Import the submodule and cache it on the current module. That is,
        # upon the next access __getattr__ will not be called.
        submodule = importlib.import_module(__name__ + "." + name)
        mod = sys.modules[__name__]
        setattr(mod, name, submodule)
        return submodule
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["z3"]

import sys
import importlib


def __getattr__(name):
    """Enable lazy loading of submodules.

    Enables `import lighthouse.tune as lh_tune; lh_tune.smt.<submodule>` with
    loading of (the submodule's heavy) depenendencies only upon being needed.
    """

    if name in __all__:
        # Import the submodule and cache it on the current module. That is,
        # upon the next access __getattr__ will not be called.
        submodule = importlib.import_module("lighthouse.tune.smt." + name)
        lighthouse_mod = sys.modules[__name__]
        setattr(lighthouse_mod, name, submodule)
        return submodule
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

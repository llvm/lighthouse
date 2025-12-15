__all__ = ["memref", "mlir", "torch"]

import sys
import importlib


def __getattr__(name):
    """Enable lazy loading of submodules.

    Enables `import lighthouse.utils as lh_utils; lh_utils.<submodule>` with
    loading of (the submodule's heavy) depenendencies only upon being needed.
    """

    if name in __all__:
        # Import the submodule and cache it on the current module. That is,
        # upon the next access __getattr__ will not be called.
        submodule = importlib.import_module("lighthouse.utils." + name)
        lighthouse_utils_mod = sys.modules[__name__]
        setattr(lighthouse_utils_mod, name, submodule)
        return submodule
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

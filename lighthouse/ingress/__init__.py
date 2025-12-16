__all__ = ["mlir_gen", "torch"]

import sys
import importlib


def __getattr__(name):
    """Enable lazy loading of submodules.

    Enables `import lighthouse.ingress as lh_ingress; lh_ingress.<submodule>` with
    loading of (the submodule's heavy) depenendencies only upon being needed.
    """

    if name in __all__:
        # Import the submodule and cache it on the current module. That is,
        # upon the next access __getattr__ will not be called.
        submodule = importlib.import_module("lighthouse.ingress." + name)
        lighthouse_ingress_mod = sys.modules[__name__]
        setattr(lighthouse_ingress_mod, name, submodule)
        return submodule
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

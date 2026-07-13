import weakref

from mlir import ir
from mlir.dialects import ext


class DialectExtension(ext.Dialect, name="base_extension"):
    """A base class for dialect extensions."""

    @classmethod
    def load(cls, *args, **kwargs):
        """Register and load the dialect into the current MLIR context.

        A Python-defined dialect is bound to the MLIR context it is loaded into,
        so it must be (re)loaded for every context in which it is used. MLIR,
        however, does not allow loading the same dialect into a context twice.
        This method makes loading idempotent per-context and derives the correct
        reload behavior automatically, so callers never need to know whether a
        reload is required:

        - Already loaded in the current context: no-op (reloading into the same
          context is unsupported and would assert).
        - Loaded in some *other* context but not this one: re-emit and replace
          the process-global registrations for the current context. This also
          covers coming back to a context after using a different one, since a
          dialect stays loaded in every context it was loaded into.
        - Never loaded: perform a fresh load.
        """
        # Per-subclass registry of the contexts this dialect is loaded in. A
        # single "last context" is insufficient: a dialect stays loaded in every
        # context it was loaded into, so interleaving contexts requires tracking
        # all of them. Contexts are held weakly so they remain garbage
        # collectable.
        loaded_contexts = cls.__dict__.get("_loaded_contexts")
        if loaded_contexts is None:
            loaded_contexts = weakref.WeakSet()
            cls._loaded_contexts = loaded_contexts

        current = ir.Context.current
        if current in loaded_contexts:
            # Already present in this context; MLIR cannot load it again here.
            return

        # If the dialect was loaded before (in any context), its process-global,
        # context-independent registrations already exist and must be replaced
        # rather than added again.
        globals_registered = hasattr(cls, "_mlir_module")

        # Registers the dialect and its op classes and loads them into the context.
        kwargs["reload"] = globals_registered
        super().load(*args, **kwargs)
        loaded_contexts.add(current)

        # Attach interfaces to just registered/loaded operations.
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()

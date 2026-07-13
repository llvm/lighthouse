from .dialect_base import DialectExtension

__all__ = ["DialectExtension"]


def register_and_load(reload: bool = False, **kwargs):
    """Register and load custom extensions into the current MLIR context.

    Loading is idempotent per-context: extensions already loaded in the current
    context are skipped, and extensions previously loaded in a different context
    are automatically re-emitted for the current one (see
    ``DialectExtension.load``). Callers therefore do not need to reason about
    whether a reload is required.

    Args:
        reload: Force a reload even if the extension was already loaded. Normally
            unnecessary as reloading is detected automatically.
        **kwargs: Additional keyword arguments to pass to the extensions' load methods.
    """
    from . import smt_ext
    from .transform import transform_ext
    from .transform import smt_ext as td_smt_ext
    from .transform import tune_ext

    dialects = [smt_ext, transform_ext, td_smt_ext, tune_ext]
    for dialect in dialects:
        dialect.register_and_load(reload=reload, **kwargs)

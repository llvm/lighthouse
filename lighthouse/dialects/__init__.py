from .dialect_base import DialectExtension

__all__ = ["DialectExtension"]


def register_and_load(reload: bool = False, **kwargs):
    """Register and load custom extensions.

    Args:
        reload: Force reload the dialects.
        **kwargs: Additional keyword arguments to pass to the dialects' load methods.
    """
    from mlir import ir
    from . import smt_ext
    from .transform import transform_ext
    from .transform import smt_ext as td_smt_ext
    from .transform import tune_ext

    dialects = [smt_ext, transform_ext, td_smt_ext, tune_ext]
    for dialect in dialects:
        if (
            not reload
            and hasattr(dialect, "_mlir_module")
            and dialect._mlir_module.context is ir.Context.current
        ):
            # Avoid reloading the dialect when possible.
            continue
        dialect.register_and_load(reload=reload, **kwargs)

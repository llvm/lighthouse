from .dialect_base import DialectExtension

__all__ = ["DialectExtension"]


def register_and_load(**kwargs):
    """Register and load custom extensions."""
    from . import smt_ext
    from .transform import transform_ext
    from .transform import smt_ext as td_smt_ext
    from .transform import tune_ext

    smt_ext.register_and_load(**kwargs)
    transform_ext.register_and_load(**kwargs)
    td_smt_ext.register_and_load(**kwargs)
    tune_ext.register_and_load(**kwargs)

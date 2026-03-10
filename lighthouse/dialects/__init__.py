__all__ = ["register_and_load", "transform_ext"]


def register_and_load(**kwargs):
    from . import transform_ext

    transform_ext.register_and_load(**kwargs)

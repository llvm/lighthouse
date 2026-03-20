def register_and_load(**kwargs):
    from .transform import transform_ext
    from .transform import smt_ext
    from .transform import tune_ext

    transform_ext.register_and_load(**kwargs)
    smt_ext.register_and_load(**kwargs)
    tune_ext.register_and_load(**kwargs)

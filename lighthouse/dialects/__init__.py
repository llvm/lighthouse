def register_and_load(**kwargs):
    from . import smt_ext
    from . import transform_ext
    from . import transform_smt_ext
    from . import transform_tune_ext

    smt_ext.register_and_load(**kwargs)
    transform_ext.register_and_load(**kwargs)
    transform_smt_ext.register_and_load(**kwargs)
    transform_tune_ext.register_and_load(**kwargs)

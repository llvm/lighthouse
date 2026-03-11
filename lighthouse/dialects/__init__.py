def register_and_load():
    from . import smt_ext
    from . import transform_smt_ext
    from . import transform_tune_ext

    smt_ext.register_and_load()
    transform_smt_ext.register_and_load()
    transform_tune_ext.register_and_load()

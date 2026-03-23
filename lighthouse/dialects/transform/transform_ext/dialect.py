from lighthouse.dialects import DialectExtension


def register_and_load(**kwargs):
    TransformExtensionDialect.load(**kwargs)


class TransformExtensionDialect(DialectExtension, name="transform_ext"):
    """A Transform Dialect extensions."""

    pass

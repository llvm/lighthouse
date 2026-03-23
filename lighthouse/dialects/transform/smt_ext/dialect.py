from lighthouse.dialects import DialectExtension


def register_and_load(**kwargs):
    TransformSMTExtensionDialect.load(**kwargs)


class TransformSMTExtensionDialect(DialectExtension, name="transform_smt_ext"):
    """A Transform Dialect extension for SMT-related operations."""

    pass

from mlir.dialects import ext


def register_and_load(**kwargs):
    """Register and load the SMTIntValue caster."""

    TransformSMTExtensionDialect.load(**kwargs)


class TransformSMTExtensionDialect(ext.Dialect, name="transform_smt_ext"):
    """A Transform Dialect extension for SMT-related operations."""

    @classmethod
    def load(cls, *args, **kwargs):
        # Registers the dialect and its op classes and loads the dialect and ops into the context.
        super().load(*args, **kwargs)

        for op in cls.operations:
            if hasattr(op, "attach_interfaces"):
                op.attach_interfaces()

from mlir.dialects import ext


class DialectExtension(ext.Dialect, name="base_extension"):
    """A base class for dialect extensions."""

    @classmethod
    def load(cls, *args, **kwargs):
        # Registers the dialect and its op classes and loads the dialect and ops into the context.
        super().load(*args, **kwargs)

        # Attach interfaces to just registered/loaded operations.
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()

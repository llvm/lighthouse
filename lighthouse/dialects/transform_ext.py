from mlir import rewrite, ir
from mlir.dialects import ext, transform


def register_and_load(context=None):
    TransformExtensionDialect.load()


class TransformExtensionDialect(ext.Dialect, name="transform_ext"):
    @classmethod
    def load(cls, *args, **kwargs):
        super().load(*args, **kwargs)
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()


class PopulatePatternOp(TransformExtensionDialect.Operation, name="populate_pattern"):
    """An operation to populate a pattern set with a specific pattern.

    To be used in the region of `transform.apply_patterns`."""

    pattern_name: ir.StringAttr
    op_kind: ir.StringAttr
    priority: ir.IntegerAttr

    # A mapping from pattern names to their corresponding rewrite functions.
    # This should be populated by the users of this operation. In effect serves
    # as a registry for rewrite patterns that can be applied by this operation.
    name_to_rewrite_pattern = {}

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.PatternDescriptorOpInterfaceModel.attach(
            cls.OPERATION_NAME, context=context
        )

    class PatternDescriptorOpInterfaceModel(transform.PatternDescriptorOpInterface):
        @staticmethod
        def populate_patterns(
            op: "PopulatePatternOp",
            patternset: rewrite.RewritePatternSet,
        ) -> None:
            patternset.add(
                op.op_kind.value,
                op.name_to_rewrite_pattern[op.pattern_name.value],
                benefit=op.priority.value,
            )


def populate_pattern(
    pattern_name: str, op_kind: str, priority: int
) -> PopulatePatternOp:
    """Camelcase constructor for PopulatePatternOp."""
    priority_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), priority)
    return PopulatePatternOp(
        op_kind=ir.StringAttr.get(op_kind),
        pattern_name=ir.StringAttr.get(pattern_name),
        priority=priority_attr,
    )

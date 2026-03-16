from mlir import rewrite, ir
from mlir.dialects import ext, transform


def register_and_load(context=None):
    """Register and load the SMTIntValue caster."""

    PatternDialect.load()


class PatternDialect(ext.Dialect, name="transform_ext"):
    pass

    @classmethod
    def load(cls, *args, **kwargs):
        super().load(*args, **kwargs)
        for op_cls in cls.operations:
            if hasattr(op_cls, "attach_interface_impls"):
                op_cls.attach_interface_impls()


class PopulatePatternOp(PatternDialect.Operation, name="populate_pattern"):
    op_kind: ir.StringAttr
    pattern_name: ir.StringAttr
    priority: ir.IntegerAttr

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
            priority = op.priority.value
            patternset.add(
                op.op_kind.value,
                op.name_to_rewrite_pattern[op.pattern_name.value],
                benefit=priority,
            )


def populate_pattern(
    op_kind: str, pattern_name: str, priority: int
) -> PopulatePatternOp:
    """Helper function to create a PopulatePatternOp."""
    priority_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), priority)
    return PopulatePatternOp(
        op_kind=ir.StringAttr.get(op_kind),
        pattern_name=ir.StringAttr.get(pattern_name),
        priority=priority_attr,
    )

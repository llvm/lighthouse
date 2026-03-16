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


# def rewrite_pattern(patterns: dict, pname: str):
#    """Return a rewrite pattern class that can be registered with MLIR.
#    The patterns dict should map op names to their corresponding match and rewrite functions."""
#
#    @ext.register_operation(PatternDialect)
#    class RewritePattern(PatternDialect.Operation, name=pname):
#        @classmethod
#        def attach_interface_impls(cls, ctx=None):
#            cls.PatternDescriptorOpInterfaceFallbackModel.attach(
#                cls.OPERATION_NAME, context=ctx
#            )
#
#        class PatternDescriptorOpInterfaceFallbackModel(
#            transform.PatternDescriptorOpInterface
#        ):
#            @staticmethod
#            def populate_patterns(
#                op: "RewritePattern",
#                patternset: rewrite.RewritePatternSet,
#            ) -> None:
#                for op_name, match_and_rewrite in patterns.items():
#                    patternset.add(op_name, match_and_rewrite, benefit=1)
#
#    return RewritePattern
#
#
# def pattern_rewrite_schedule(patterns: dict, pname: str = "rewrite_pattern"):
#    """Return a transform module that applies the given rewrite patterns.
#    patterns: dict mapping op names to match-and-rewrite functions.
#    pname: name for the generated rewrite pattern operation."""
#
#    rw_pattern = rewrite_pattern(patterns, pname)
#    PatternDialect.load(register=False, reload=False)
#    rw_pattern.attach_interface_impls()
#
#    with schedule_boilerplate() as (schedule, named_seq):
#        apply_patterns_op = transform.ApplyPatternsOp(named_seq.bodyTarget)
#        with ir.InsertionPoint(apply_patterns_op.patterns):
#            rw_pattern()
#        transform.yield_([named_seq.bodyTarget])
#        named_seq.verify()
#
#    schedule.body.operations[0].verify()
#    return schedule

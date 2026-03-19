from mlir import rewrite, ir
from mlir.dialects import ext, transform
from lighthouse.schedule import schedule_boilerplate


@ext.register_dialect
class PatternDialect(ext.Dialect, name="lighthouse"):
    pass


def rewrite_pattern(patterns: dict, pname: str):
    """Return a rewrite pattern class that can be registered with MLIR.
    The patterns dict should map op names to their corresponding match and rewrite functions."""

    @ext.register_operation(PatternDialect, replace=True)
    class RewritePattern(PatternDialect.Operation, name=pname):
        @classmethod
        def attach_interface_impls(cls, ctx=None):
            cls.PatternDescriptorOpInterfaceFallbackModel.attach(
                cls.OPERATION_NAME, context=ctx
            )

        class PatternDescriptorOpInterfaceFallbackModel(
            transform.PatternDescriptorOpInterface
        ):
            @staticmethod
            def populate_patterns(
                op: "RewritePattern",
                patternset: rewrite.RewritePatternSet,
            ) -> None:
                for op_name, match_and_rewrite in patterns.items():
                    patternset.add(op_name, match_and_rewrite, benefit=1)

    return RewritePattern


def pattern_rewrite_schedule(patterns: dict, pname: str = "rewrite_pattern"):
    """Return a transform module that applies the given rewrite patterns.
    patterns: dict mapping op names to match-and-rewrite functions.
    pname: name for the generated rewrite pattern operation."""

    rw_pattern = rewrite_pattern(patterns, pname)
    PatternDialect.load(register=False, reload=False)
    rw_pattern.attach_interface_impls()

    with schedule_boilerplate() as (schedule, named_seq):
        apply_patterns_op = transform.ApplyPatternsOp(named_seq.bodyTarget)
        with ir.InsertionPoint(apply_patterns_op.patterns):
            rw_pattern()
        transform.yield_([named_seq.bodyTarget])
        named_seq.verify()

    schedule.body.operations[0].verify()
    return schedule

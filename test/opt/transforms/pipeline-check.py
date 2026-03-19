from mlir import ir
from mlir.dialects import transform
from lighthouse.pipeline.helper import match
from lighthouse.pipeline.opt import PassBundles, apply_bundle


def create_schedule(options: dict = {}) -> ir.Module:
    """Creates a Transform Schedule for the test's optimization pipeline."""

    schedule_module = ir.Module.create()
    schedule_module.operation.attributes["transform.with_named_sequence"] = (
        ir.UnitAttr.get()
    )
    with ir.InsertionPoint(schedule_module.body):
        named_sequence = transform.named_sequence(
            "__transform_main",
            [transform.AnyOpType.get()],
            [],
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            anytype = transform.AnyOpType.get()
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            mod = transform.get_parent_op(
                anytype,
                func,
                op_name="builtin.module",
                deduplicate=True,
            )
            mod = apply_bundle(mod, PassBundles["BufferizationBundle"])
            mod = apply_bundle(mod, PassBundles["MLIRLoweringBundle"])
            mod = apply_bundle(mod, PassBundles["CleanupBundle"])

            if not options.get("skip_llvm", False):
                mod = apply_bundle(mod, PassBundles["LLVMLoweringBundle"])
            transform.YieldOp()

    return schedule_module

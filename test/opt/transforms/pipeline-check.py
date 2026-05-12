from mlir import ir
from mlir.dialects import transform
from lighthouse.pipeline.helper import apply_registered_pass, match
from lighthouse.pipeline.stage import apply_bundle


def create_schedule(skip_llvm=False) -> ir.Module:
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
            mod = apply_bundle(mod, "bufferization.yaml")
            mod = apply_bundle(mod, "bufferization_cleanup.yaml")
            mod = apply_registered_pass(mod, "convert-linalg-to-loops")
            mod = apply_bundle(mod, "cleanup.yaml")

            if not skip_llvm:
                mod = apply_bundle(mod, "llvm_lowering.yaml")
            transform.YieldOp()

    return schedule_module

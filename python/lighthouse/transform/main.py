from mlir import ir
from mlir.dialects.transform import interpreter as transform_interpreter


def apply(schedule: ir.Operation | ir.OpView, payload: ir.Module) -> None:
    assert schedule.parent and "transform.with_named_sequence" in schedule.parent.attributes
    assert "transform.with_named_sequence" in schedule.parent.attributes
    assert isinstance(schedule.parent.attributes["transform.with_named_sequence"], ir.UnitAttr)

    transform_interpreter.apply_named_sequence(
        payload_root=payload,
        transform_root=schedule,
        transform_module=schedule.parent,
    )

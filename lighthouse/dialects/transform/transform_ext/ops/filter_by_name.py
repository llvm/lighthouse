from mlir import ir
from mlir.dialects import transform

from lighthouse.dialects.transform.transform_ext.utils.make_filter_handles_op import (
    make_filter_handles_op,
)


def _normalize_op_names(op_names: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a list of operation names into a canonical tuple."""
    if isinstance(op_names, str):
        op_names = [op_names]
    names = tuple(dict.fromkeys(op_names))
    if not names:
        raise ValueError("op_names must contain at least one operation name")
    return names


def _parse_op_names(params) -> frozenset[str] | None:
    if len(params) != 1 or not isinstance(params[0], ir.ArrayAttr):
        return None
    names = set()
    for name_attr in params[0]:
        if not isinstance(name_attr, ir.StringAttr):
            return None
        names.add(name_attr.value)
    return frozenset(names)


def _name_matches(op: ir.Operation | ir.OpView, allowed_names: frozenset[str]) -> bool:
    return op.name in allowed_names


FilterByNameOp = make_filter_handles_op(
    "filter_by_name",
    _name_matches,
    parse_param=_parse_op_names,
)


def filter_by_name(
    target: ir.Value[transform.AnyOpType],
    op_names: str | list[str] | tuple[str, ...],
) -> ir.Value:
    """Return the subset of `target` whose payload operation names match `op_names`."""
    names = _normalize_op_names(op_names)
    names_attr = ir.ArrayAttr.get([ir.StringAttr.get(name) for name in names])
    op_names_param = transform.ParamConstantOp(transform.AnyParamType.get(), names_attr)
    return FilterByNameOp(target=target, param=op_names_param).ops

"""IR rewrites used by dependent-reduction fusion."""

from mlir import ir
from mlir.dialects import arith, tensor

from lighthouse.utils.mlir import (
    cast_float,
    clone_block_body,
    defining_op,
    float_width,
    op_attributes,
    opview,
    project_dims,
    remap_dims,
)

__all__ = [
    "cast_float",
    "clone_block_body",
    "clone_op_deep_with_map",
    "clone_op_with_map",
    "depends_on_op",
    "mixed_sizes",
    "move_value_definitions",
    "op_attributes",
    "operand_eliminating_constant",
    "project_dims",
    "remap_dims",
    "wider_float_type",
]


def clone_op_with_map(
    op: ir.Operation | ir.OpView,
    value_map: dict,
    *,
    result_type: ir.Type,
) -> ir.Operation:
    """Clone a region-free scalar op at `result_type` with mapped operands."""
    ov = opview(op)
    assert not any(len(r.blocks) for r in ov.operation.regions)
    attributes = op_attributes(ov)
    value = attributes.get("value")
    if isinstance(value, ir.FloatAttr):
        attributes["value"] = ir.FloatAttr.get(result_type, ir.FloatAttr(value).value)
    cloned = ir.Operation.create(
        ov.operation.name,
        results=[result_type] * len(ov.results),
        operands=[value_map.get(o, o) for o in ov.operands],
        attributes=attributes,
    )
    value_map.update(zip(ov.results, cloned.results))
    return cloned


def clone_op_deep_with_map(
    op: ir.Operation | ir.OpView, value_map: dict
) -> ir.Operation:
    """Clone an op with regions, remapping operands and captured values."""
    ov = opview(op)
    cloned = ov.operation.clone()
    for i, operand in enumerate(ov.operands):
        if operand in value_map:
            cloned.operands[i] = value_map[operand]

    def remap_captured(inner: ir.Operation) -> ir.WalkResult:
        for i, operand in enumerate(inner.operands):
            if operand in value_map:
                inner.operands[i] = value_map[operand]
        return ir.WalkResult.ADVANCE

    cloned.walk(remap_captured)
    value_map.update(zip(ov.results, cloned.results))
    return cloned


def _ancestor_in_block(op: ir.Operation | ir.OpView, block: ir.Block):
    """The ancestor of `op` that sits directly in `block`, or None."""
    cur = opview(op).operation
    while cur is not None:
        parent_block = cur.block
        if parent_block is None:
            return None
        if parent_block == block:
            return cur
        owner = parent_block.owner
        cur = owner.operation if owner is not None else None
    return None


def properly_dominates(
    a: ir.Operation | ir.OpView, b: ir.Operation | ir.OpView
) -> bool:
    """Whether A precedes B in their shared block or encloses B."""
    a_op, b_op = opview(a).operation, opview(b).operation
    if a_op == b_op:
        return False
    block = a_op.block
    if block is None:
        return False
    b_anchor = _ancestor_in_block(b_op, block)
    if b_anchor is None:
        return False
    if b_anchor == a_op:
        # `a` encloses `b`.
        return True
    return a_op.is_before_in_block(b_anchor)


def move_value_definitions(
    values: list[ir.Value], before_op: ir.Operation | ir.OpView
) -> bool:
    """Move needed definitions before `before_op` in program order.

    Return False if a definition must move across blocks.
    """
    anchor = opview(before_op).operation
    block = anchor.block
    to_move: dict = {}
    stack = [defining_op(v) for v in values]
    while stack:
        cur = stack.pop()
        if cur is None or cur.__hash__() in to_move:
            continue
        if properly_dominates(cur, anchor):
            continue
        if cur.block != block:
            return False
        to_move[cur.__hash__()] = cur
        for operand in cur.operands:
            producer = defining_op(operand)
            if producer is not None:
                stack.append(producer)

    if not to_move:
        return True
    # Program order within the block, so relative order survives the move.
    ordered = [op for op in block.operations if op.operation.__hash__() in to_move]
    for op in ordered:
        op.operation.move_before(anchor)
    return True


def backward_slice(value: ir.Value) -> set:
    """Collect hashes of operations that transitively define `value`."""
    slice_ops: set = set()
    def_op = defining_op(value)
    if def_op is None:
        return slice_ops
    stack = [def_op]
    while stack:
        cur = stack.pop()
        key = cur.__hash__()
        if key in slice_ops:
            continue
        slice_ops.add(key)
        for operand in cur.operands:
            producer = defining_op(operand)
            if producer is not None:
                stack.append(producer)
    return slice_ops


def depends_on_op(value: ir.Value, op: ir.Operation | ir.OpView) -> bool:
    """Whether `value` depends on `op` and cannot be hoisted above it."""
    target = opview(op).operation
    def_op = defining_op(value)
    if def_op is not None and def_op == target:
        return True
    return target.__hash__() in backward_slice(value)


def wider_float_type(a: ir.Type, b: ir.Type) -> ir.Type | None:
    """Choose the wider float type; reject f16/bf16 mixed formats."""
    width_a, width_b = float_width(a), float_width(b)
    if width_a is None or width_b is None:
        return None
    if a == b:
        return a
    if width_a == width_b:
        return None
    return a if width_a > width_b else b


def operand_eliminating_constant(
    op: ir.Operation | ir.OpView, element_type: ir.Type
) -> ir.Attribute | None:
    """Return 0 for additive ops or 1 for multiplicative ops, else None.

    These stand-ins need not preserve an op's value (``0 - x`` and ``1 / x``).
    The data factor cancels in the correction ratio when E is separable.
    """
    ov = opview(op)
    if isinstance(ov, (arith.AddFOp, arith.SubFOp)):
        return ir.FloatAttr.get(element_type, 0.0)
    if isinstance(ov, (arith.MulFOp, arith.DivFOp)):
        return ir.FloatAttr.get(element_type, 1.0)
    if not isinstance(element_type, (ir.IntegerType, ir.IndexType)):
        return None
    if isinstance(ov, (arith.AddIOp, arith.SubIOp)):
        return ir.IntegerAttr.get(element_type, 0)
    if isinstance(ov, arith.MulIOp):
        return ir.IntegerAttr.get(element_type, 1)
    return None


def mixed_sizes(value: ir.Value) -> list:
    """Return static extents as ints and dynamic extents as ``tensor.dim``."""
    shaped = ir.ShapedType(value.type)
    sizes = []
    for pos, extent in enumerate(shaped.shape):
        if ir.ShapedType.is_dynamic_size(extent):
            index = arith.constant(ir.IndexType.get(), pos)
            sizes.append(tensor.dim(value, index))
        else:
            sizes.append(extent)
    return sizes

"""Partition a function body into tagged subgraphs.

A subgraph is a set of ops that is lowered together, e.g. into one workgroup
kernel. Every op of a subgraph carries its kind and a function-unique id, and
exactly one op, the anchor, is marked as the op to tile. The remaining members
are either producers of the anchor (fused into its loop) or consumers of it
(fused forward into its loop).

Matchers run in a fixed order and only claim untagged ops, so the more
demanding kinds (e.g. GEMMs) take precedence over generic elementwise groups.
"""

from collections.abc import Callable, Iterable

from mlir import ir
from mlir.dialects import tensor

from lighthouse.dialects.transform.transform_ext.ops.filter_elementwise import (
    is_elementwise,
)
from lighthouse.utils.mlir import opview

SUBGRAPH_KIND_ATTR_NAME = "transform_ext.subgraph_kind"
SUBGRAPH_ID_ATTR_NAME = "transform_ext.subgraph_id"
SUBGRAPH_ANCHOR_ATTR_NAME = "transform_ext.subgraph_anchor"

_DATA_MOVEMENT_OPS = (
    tensor.ExpandShapeOp,
    tensor.CollapseShapeOp,
    tensor.ExtractSliceOp,
    tensor.ConcatOp,
)


def subgraph_id(op: ir.Operation | ir.OpView) -> int | None:
    attrs = opview(op).operation.attributes
    if SUBGRAPH_ID_ATTR_NAME not in attrs:
        return None
    return ir.IntegerAttr(attrs[SUBGRAPH_ID_ATTR_NAME]).value


def is_tagged(op: ir.Operation | ir.OpView) -> bool:
    return subgraph_id(op) is not None


def tag(ops: Iterable, anchor, kind: str, id: int) -> None:
    i64 = ir.IntegerType.get_signless(64)
    for op in ops:
        attrs = opview(op).operation.attributes
        attrs[SUBGRAPH_KIND_ATTR_NAME] = ir.StringAttr.get(kind)
        attrs[SUBGRAPH_ID_ATTR_NAME] = ir.IntegerAttr.get(i64, id)
    opview(anchor).operation.attributes[SUBGRAPH_ANCHOR_ATTR_NAME] = ir.UnitAttr.get()


def clear_tags(op: ir.Operation | ir.OpView) -> None:
    def clear(nested):
        attrs = nested.attributes
        for name in (
            SUBGRAPH_KIND_ATTR_NAME,
            SUBGRAPH_ID_ATTR_NAME,
            SUBGRAPH_ANCHOR_ATTR_NAME,
        ):
            if name in attrs:
                del attrs[name]
        return ir.WalkResult.ADVANCE

    opview(op).operation.walk(clear)


def _producer(value: ir.Value) -> ir.Operation | None:
    owner = value.owner
    if isinstance(owner, ir.OpView):
        return owner.operation
    if isinstance(owner, ir.Operation):
        return owner
    return None


def _users(op: ir.Operation) -> list[ir.Operation]:
    return [use.owner.operation for result in op.results for use in result.uses]


def _is_linalg(op: ir.Operation) -> bool:
    return op.name.startswith("linalg.")


def _is_elementwise_linalg(op: ir.Operation) -> bool:
    return _is_linalg(op) and is_elementwise(op)


def _clone_for(producer: ir.Operation, members: list) -> ir.Operation:
    """Clone `producer` and redirect the uses by `members` to the clone."""
    with ir.InsertionPoint(producer):
        clone = producer.clone()
    for old, new in zip(producer.results, clone.results):
        for use in [u for u in old.uses if u.owner.operation in members]:
            use.owner.operands[use.operand_number] = new
    return clone


def _match_gemm(op: ir.Operation, _block_ops) -> list[tuple[list, ir.Operation]]:
    """GEMM subgraph: matmul, its single-use elementwise epilogue chain, and the
    elementwise producers (fill, transposes, broadcasts, ...) feeding it.
    Producers that also have users outside the subgraph are cloned, i.e.
    recomputed per subgraph as tile-and-fuse would do. The anchor is the last
    op of the epilogue chain."""
    if op.name != "linalg.matmul" or is_tagged(op):
        return []
    members = [op]
    anchor = op
    while len(anchor.results) == 1:
        uses = list(anchor.results[0].uses)
        if len(uses) != 1:
            break
        user = uses[0].owner.operation
        if is_tagged(user) or not _is_elementwise_linalg(user):
            break
        members.append(user)
        anchor = user

    worklist = list(members)
    while worklist:
        current = worklist.pop()
        for value in current.operands:
            producer = _producer(value)
            if (
                producer is None
                or producer in members
                or is_tagged(producer)
                or not _is_elementwise_linalg(producer)
            ):
                continue
            if any(user not in members for user in _users(producer)):
                producer = _clone_for(producer, members)
            members.append(producer)
            worklist.append(producer)
    return [(members, anchor)]


def _forward_closure(root: ir.Operation, members: list) -> list:
    reached = [root]
    worklist = [root]
    while worklist:
        for user in _users(worklist.pop()):
            if user in members and user not in reached:
                reached.append(user)
                worklist.append(user)
    return reached


def _match_elementwise(
    op: ir.Operation, block_ops: list
) -> list[tuple[list, ir.Operation]]:
    """Elementwise subgraphs: connected components of untagged elementwise
    linalg ops and the static data movement ops between them (reshapes, slices,
    concats). The anchor is the first linalg op from which all linalg members
    are reachable; the remaining ops are fused forward into its loop. Components
    without such an op are left untagged."""

    def candidate(o):
        return not is_tagged(o) and (
            _is_elementwise_linalg(o) or isinstance(o.opview, _DATA_MOVEMENT_OPS)
        )

    if not candidate(op):
        return []
    component = [op]
    worklist = [op]
    while worklist:
        current = worklist.pop()
        neighbors = _users(current) + [
            p for p in map(_producer, current.operands) if p is not None
        ]
        for neighbor in neighbors:
            if neighbor not in component and candidate(neighbor):
                component.append(neighbor)
                worklist.append(neighbor)
    order = {o: i for i, o in enumerate(block_ops)}
    if any(o not in order for o in component):
        return []
    component.sort(key=order.__getitem__)
    linalg_ops = [o for o in component if _is_linalg(o)]
    for anchor in linalg_ops:
        reached = _forward_closure(anchor, component)
        if all(o in reached for o in linalg_ops):
            return [(component, anchor)]
    return []


SUBGRAPH_MATCHERS: dict[str, Callable] = {
    "gemm": _match_gemm,
    "elementwise": _match_elementwise,
}


def tag_subgraphs(block: ir.Block, kinds: Iterable[str]) -> list[ir.Operation]:
    """Tag the subgraphs of `kinds` (in order) among the ops of `block`.

    Returns the linalg ops of `block` that are left untagged."""
    next_id = 1 + max(
        (subgraph_id(o) or 0 for o in block.operations),
        default=0,
    )
    for kind in kinds:
        matcher = SUBGRAPH_MATCHERS[kind]
        for op in list(block.operations):
            block_ops = [o.operation for o in block.operations]
            for members, anchor in matcher(op.operation, block_ops):
                tag(members, anchor, kind, next_id)
                next_id += 1
    return [
        o.operation
        for o in block.operations
        if _is_linalg(o.operation) and not is_tagged(o)
    ]


def subgraph_members(anchor: ir.Operation) -> tuple[list, list]:
    """Return the (producers, consumers) of a subgraph anchor in program order.

    Producers are the linalg members the anchor depends on, consumers are the
    members that depend on the anchor. Other members, e.g. reshapes feeding the
    anchor, stay outside of the anchor's loop."""
    id = subgraph_id(anchor)
    members = [
        o.operation
        for o in anchor.block.operations
        if o.operation != anchor and subgraph_id(o) == id
    ]
    consumers = _forward_closure(anchor, members + [anchor])[1:]
    reached = {anchor}
    for o in reversed(members):
        if any(u in reached for u in _users(o)):
            reached.add(o)
    producers = [o for o in members if o in reached and _is_linalg(o)]
    order = {o: i for i, o in enumerate(members)}
    consumers.sort(key=order.__getitem__)
    return producers, consumers


__all__ = [
    "SUBGRAPH_ANCHOR_ATTR_NAME",
    "SUBGRAPH_ID_ATTR_NAME",
    "SUBGRAPH_KIND_ATTR_NAME",
    "SUBGRAPH_MATCHERS",
    "clear_tags",
    "is_tagged",
    "subgraph_id",
    "subgraph_members",
    "tag_subgraphs",
]

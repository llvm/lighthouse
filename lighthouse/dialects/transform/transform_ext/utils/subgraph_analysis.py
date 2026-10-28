"""Partition a function body into tagged subgraphs.

A subgraph is a set of ops that is lowered together, e.g. into one workgroup
kernel. Every op of a subgraph carries its kind and a function-unique id, and
exactly one op, the anchor, is marked as the op to tile. The remaining members
are either producers of the anchor (fused into its loop) or consumers of it
(fused forward into its loop).

Matchers run in a fixed order and only claim untagged ops, so the more
demanding kinds (e.g. GEMMs) take precedence over generic elementwise groups.
They never modify IR. An optional claim predicate lets preparation reserve
candidate groups without attaching attributes; tagging uses the same matchers
after preparation has isolated shared producers.
"""

from collections.abc import Callable, Iterable

from mlir import ir
from mlir.dialects import tensor

from lighthouse.dialects.transform.transform_ext.ops.filter_elementwise import (
    is_elementwise,
)
from lighthouse.dialects.transform.transform_ext.ops.filter_reduction_ops import (
    is_reduction_op,
)
from lighthouse.utils.mlir import opview
from lighthouse.dialects.transform.transform_ext.ops.filter_contraction_ops import (
    is_contraction_op,
)
from lighthouse.dialects.transform.transform_ext.utils.ir_rewrite import (
    _ancestor_in_block,
)

SUBGRAPH_KIND_ATTR_NAME = "transform_ext.subgraph_kind"
SUBGRAPH_ID_ATTR_NAME = "transform_ext.subgraph_id"
SUBGRAPH_ANCHOR_ATTR_NAME = "transform_ext.subgraph_anchor"
SUBGRAPH_EPILOGUE_ATTR_NAME = "transform_ext.subgraph_epilogue"

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
    ops = [opview(op).operation for op in ops]
    anchor = opview(anchor).operation
    i64 = ir.IntegerType.get_signless(64)
    for op in ops:
        attrs = opview(op).operation.attributes
        attrs[SUBGRAPH_KIND_ATTR_NAME] = ir.StringAttr.get(kind)
        attrs[SUBGRAPH_ID_ATTR_NAME] = ir.IntegerAttr.get(i64, id)
    anchor.attributes[SUBGRAPH_ANCHOR_ATTR_NAME] = ir.UnitAttr.get()
    if kind == "attention":
        consumers = _forward_closure(anchor, ops)[1:]
        # A plain result reshape is handled by reshaping the destination.
        # A transpose/layout epilogue must instead be fused into the WG.
        if any(
            consumer.name not in ("tensor.expand_shape", "tensor.collapse_shape")
            for consumer in consumers
        ):
            for consumer in consumers:
                consumer.attributes[SUBGRAPH_EPILOGUE_ATTR_NAME] = ir.UnitAttr.get()


def clear_tags(op: ir.Operation | ir.OpView) -> None:
    def clear(nested):
        attrs = nested.attributes
        for name in (
            SUBGRAPH_KIND_ATTR_NAME,
            SUBGRAPH_ID_ATTR_NAME,
            SUBGRAPH_ANCHOR_ATTR_NAME,
            SUBGRAPH_EPILOGUE_ATTR_NAME,
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


def _match_gemm(
    op: ir.Operation, block_ops, is_claimed=is_tagged
) -> list[tuple[list, ir.Operation]]:
    """GEMM subgraph: matmul, its single-use elementwise epilogue chain, and the
    elementwise producers (fill, transposes, broadcasts, ...) feeding it.
    Producers are collected without cloning or redirecting uses. The anchor is
    the last op of the epilogue chain. Token-only expansions and elementwise
    epilogue DAGs are also claimed forward. At a join, the last available branch owns the
    epilogue: external computations must precede the root matmul. Local
    elementwise setup may follow it if its inputs are available. This
    keeps parallel projections separate and their join before the next GEMM.
    Collapses into a following GEMM remain outside to avoid tile-local
    reshape copies during bufferization."""
    if op.name != "linalg.matmul" or is_claimed(op):
        return []
    members = [op]
    anchor = op

    root_position = block_ops.index(op)

    def available_input(value):
        producer = _producer(value)
        if (
            producer is None
            or producer in members
            or producer not in block_ops
            or block_ops.index(producer) <= root_position
        ):
            return True
        if not is_claimed(producer) and _is_elementwise_linalg(producer):
            inputs = (
                producer.opview.inputs
                if producer.name == "linalg.generic"
                else producer.operands[:-1]
            )
            return all(available_input(operand) for operand in inputs)
        return False

    def available(user):
        inputs = user.opview.inputs if user.name == "linalg.generic" else user.operands
        return all(available_input(value) for value in inputs)

    while len(anchor.results) == 1:
        uses = list(anchor.results[0].uses)
        if len(uses) != 1:
            break
        user = uses[0].owner.operation
        if (
            user not in block_ops
            or is_claimed(user)
            or not _is_elementwise_linalg(user)
            or not available(user)
        ):
            break
        members.append(user)
        anchor = user

    for user in block_ops[block_ops.index(anchor) + 1 :]:
        if is_claimed(user) or not any(
            _producer(value) in members
            for value in user.operands
            if _producer(value) is not None
        ):
            continue
        if _is_elementwise_linalg(user) and user.name == "linalg.generic":
            eligible = True
        elif isinstance(user.opview, tensor.ExpandShapeOp):
            # Only reshape token dimensions; head/column splitting remains a
            # separate subgraph (e.g. projection -> RoPE).
            groups = [
                [int(dim) for dim in group] for group in user.opview.reassociation
            ]
            eligible = groups[-1] == [max(dim for group in groups for dim in group)]
        else:
            eligible = False
        if eligible and available(user):
            members.append(user)

    worklist = list(members)
    while worklist:
        current = worklist.pop()
        for value in current.operands:
            producer = _producer(value)
            if (
                producer is None
                or producer not in block_ops
                or producer in members
                or is_claimed(producer)
                or not _is_elementwise_linalg(producer)
            ):
                continue
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
    op: ir.Operation, block_ops: list, is_claimed=is_tagged
) -> list[tuple[list, ir.Operation]]:
    """Elementwise subgraphs: connected components of untagged elementwise
    linalg ops and the static data movement ops between them (reshapes, slices,
    concats). The anchor is the first linalg op from which all linalg members
    are reachable; the remaining ops are fused forward into its loop. Components
    without such an op are left untagged."""

    def candidate(o):
        return (
            o in block_ops
            and not is_claimed(o)
            and (_is_elementwise_linalg(o) or isinstance(o.opview, _DATA_MOVEMENT_OPS))
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


def _match_reduction(
    op: ir.Operation, block_ops: list, is_claimed=is_tagged
) -> list[tuple[list, ir.Operation]]:
    """Claim normalization without crossing shared input materializations.

    Follow reduction consumers first, then absorb producers only when all their
    uses are already local. A preceding residual also used after normalization
    stays outside; the square/cast chain used only by normalization stays inside.
    Shared accumulator fills are included for preparation to isolate.
    """
    if op.name != "linalg.generic" or not is_reduction_op(op) or is_claimed(op):
        return []

    def candidate(value):
        return (
            value in block_ops
            and not is_claimed(value)
            and (_is_elementwise_linalg(value) or value == op)
        )

    members = _forward_closure(op, [value for value in block_ops if candidate(value)])
    changed = True
    while changed:
        changed = False
        for current in list(members):
            for value in current.operands:
                producer = _producer(value)
                if (
                    producer is not None
                    and producer not in members
                    and candidate(producer)
                    and (
                        all(user in members for user in _users(producer))
                        or producer.name == "linalg.fill"
                    )
                ):
                    members.append(producer)
                    changed = True
    sinks = [
        member for member in members if not any(u in members for u in _users(member))
    ]
    if len(sinks) != 1 or not _is_elementwise_linalg(sinks[0]):
        return []
    anchor = sinks[0]
    if any(
        u not in members
        for member in members
        if member != anchor and member.name != "linalg.fill"
        for u in _users(member)
    ):
        return []
    order = {value: i for i, value in enumerate(block_ops)}
    members.sort(key=order.__getitem__)
    return [(members, anchor)]


def _contains(op, names):
    found = []

    def visit(inner):
        if inner.name in names:
            found.append(inner)
        return ir.WalkResult.ADVANCE

    op.walk(visit)
    return bool(found)


def _match_attention(root, block_ops, is_claimed=is_tagged):
    """Claim a closed QK-softmax-PV DAG, stopping at projection/RoPE inputs."""
    if (
        is_claimed(root)
        or root.name not in ("linalg.generic", "linalg.batch_matmul")
        or not is_contraction_op(root)
    ):
        return []
    movement = {
        "tensor.expand_shape",
        "tensor.collapse_shape",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "linalg.transpose",
    }
    members, worklist = [root], [root]
    while worklist:
        for user in _users(worklist.pop()):
            if (
                user in block_ops
                and user not in members
                and not is_claimed(user)
                and (
                    (_is_linalg(user) and user.name != "linalg.matmul")
                    or user.name in movement
                )
            ):
                members.append(user)
                worklist.append(user)
    if sum(is_contraction_op(o) for o in members) != 2:
        return []
    if not any(_contains(o, {"math.exp"}) for o in members) or not any(
        _contains(o, {"arith.maximumf", "arith.maxnumf"}) for o in members
    ):
        return []
    sinks = [o for o in members if not any(u in members for u in _users(o))]
    if len(sinks) != 1 or len(sinks[0].results) != 1:
        return []
    sink = sinks[0]
    if any(u not in members for o in members if o != sink for u in _users(o)):
        return []

    def prologue(op):
        if is_claimed(op):
            return False
        if op.name in movement or op.name in (
            "tensor.empty",
            "arith.constant",
            "linalg.fill",
        ):
            return True
        if not _is_elementwise_linalg(op):
            return False
        if _contains(op, {"arith.addf", "arith.subf", "arith.negf"}):
            return False

        def from_matmul(value):
            producer = _producer(value)
            while producer is not None and producer.name in (
                "tensor.expand_shape",
                "tensor.collapse_shape",
            ):
                producer = _producer(producer.operands[0])
            return producer is not None and producer.name == "linalg.matmul"

        if any(from_matmul(value) for value in op.operands):
            return False
        return not _contains(op, {"arith.mulf"}) or len(op.opview.inputs) <= 1

    worklist = list(members)
    while worklist:
        current = worklist.pop()
        operands = []

        def captures(inner):
            operands.extend(inner.operands)
            return ir.WalkResult.ADVANCE

        current.walk(captures)
        for value in operands:
            producer = _producer(value)
            if (
                producer is not None
                and producer in block_ops
                and producer not in members
                and prologue(producer)
            ):
                members.append(producer)
                worklist.append(producer)
    members.sort(key=block_ops.index)
    anchor = sink
    while anchor.name in movement:
        source = _producer(anchor.operands[0])
        if source is None or source not in members:
            break
        anchor = source
    return [(members, anchor)]


SUBGRAPH_MATCHERS: dict[str, Callable] = {
    "attention": _match_attention,
    "reduction": _match_reduction,
    "gemm": _match_gemm,
    "elementwise": _match_elementwise,
}


def tag_subgraphs(block: ir.Block, kinds: Iterable[str]) -> list[ir.Operation]:
    """Tag the subgraphs of `kinds` (in order) among the ops of `block`.

    Only attributes are changed. Shared producers must have been isolated by
    preparation. Returns unsupported/shared ops without tagging if isolation is
    needed; otherwise returns the linalg ops left untagged."""
    block_ops = [op.operation for op in block.operations]
    claimed = {op for op in block_ops if is_tagged(op)}
    groups = []
    for kind in kinds:
        for op in block_ops:
            for members, anchor in SUBGRAPH_MATCHERS[kind](
                op, block_ops, claimed.__contains__
            ):
                shared = (
                    shared_producers(members, anchor) if kind != "elementwise" else []
                )
                if shared:
                    return shared
                groups.append((members, anchor, kind))
                claimed.update(members)
    next_id = 1 + max(
        (subgraph_id(o) or 0 for o in block.operations),
        default=0,
    )
    for members, anchor, kind in groups:
        tag(members, anchor, kind, next_id)
        next_id += 1
    return [
        o.operation
        for o in block.operations
        if _is_linalg(o.operation) and not is_tagged(o)
    ]


def shared_producers(members: list, anchor: ir.Operation) -> list[ir.Operation]:
    """External-use members feeding the anchor or its forward consumers.

    The anchor and forward consumers may expose additional results; those uses
    do not request producer recomputation.
    """
    consumers = set(_forward_closure(anchor, members))
    return [
        op
        for op in members
        if op not in consumers
        and any(
            _ancestor_in_block(user, anchor.block) not in members for user in _users(op)
        )
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
    "shared_producers",
    "subgraph_id",
    "subgraph_members",
    "tag_subgraphs",
]

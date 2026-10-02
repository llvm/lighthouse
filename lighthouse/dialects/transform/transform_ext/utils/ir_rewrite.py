"""IR analysis helpers used by dependent-reduction legality."""

from mlir import ir
from mlir.dialects import arith, linalg, tensor

from lighthouse.utils.mlir import defining_op, opview

__all__ = [
    "constant_int_value",
    "float_width",
    "is_defined_as_zero",
    "match_reduction",
    "post_dominates",
    "resolve_slice_source",
    "wider_float_type",
]


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


def post_dominates(a: ir.Operation | ir.OpView, b: ir.Operation | ir.OpView) -> bool:
    """Whether `a` post-dominates `b`, for ops related through one block.

    Within a single block with no control flow, post-dominance is the reverse of
    program order: `a` post-dominates `b` when `a` comes at or after `b`. Returns
    False when no common block is found (the conservative answer).
    """
    a_op, b_op = opview(a).operation, opview(b).operation
    if a_op == b_op:
        return True
    block = a_op.block
    if block is None:
        return False
    b_anchor = _ancestor_in_block(b_op, block)
    if b_anchor is None:
        return False
    if b_anchor == a_op:
        return True
    return b_anchor.is_before_in_block(a_op)


def backward_slice(value: ir.Value) -> set:
    """The ops transitively producing `value`, as a set of operation hashes.

    A plain DFS over operands, standing in for ``getBackwardSlice``. Regions are
    traversed only through their ops' operands, which suffices for the
    straight-line tensor IR the fusion inspects.
    """
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


def resolve_slice_source(value: ir.Value) -> ir.Value:
    """Resolve `value` through any chain of ``tensor.extract_slice`` to its source.

    Inside a tiled reduction loop the inner reduction reads slices of the real
    input tensors, so comparing its inputs against an untiled op's inputs means
    looking through those tile slices.
    """
    while True:
        def_op = defining_op(value)
        if def_op is None or not isinstance(def_op.opview, tensor.ExtractSliceOp):
            return value
        value = def_op.opview.source


def match_reduction(
    iter_carried_args: list[ir.BlockArgument], red_pos: int
) -> tuple[ir.Value | None, list]:
    """Match a generic reduction, returning ``(reduced_value, combiner_ops)``.

    A port of ``mlir::matchReduction``. Relies on the same invariants: the first
    combiner is a binary op taking the iteration-carried value and the reduced
    value; the def-use chain from it is single-use, side-effect free and
    immediately nested in the reduction region; and it ends at the terminator.
    Returns ``(None, [])`` when no reduction is matched.

    Matching is limited to a single combiner op, as upstream does.
    """
    combiners: list = []
    carried = iter_carried_args[red_pos]
    uses = list(carried.uses)
    if len(uses) != 1:
        return None, []

    combiner = uses[0].owner.operation
    if len(combiner.operands) != 2:
        return None, []
    reduced = (
        combiner.operands[1]
        if combiner.operands[0] == carried
        else combiner.operands[0]
    )

    # The reduced value must not itself depend on a carried value, or the chain
    # is not a plain accumulate.
    region_block = carried.owner
    carried_set = set(iter_carried_args)
    if reduced in carried_set:
        return None, []
    slice_ops = backward_slice(reduced)
    if any(
        operand in carried_set
        for op in region_block.operations
        if op.operation.__hash__() in slice_ops
        for operand in op.operands
    ):
        return None, []

    # Walk the def-use chain to the terminator, gathering combiners in order.
    while not combiner.has_trait(ir.IsTerminatorTrait):
        if len(combiner.results) != 1:
            return None, []
        combiner_uses = list(combiner.results[0].uses)
        if len(combiner_uses) != 1:
            return None, []
        if combiner.block != region_block:
            return None, []
        combiners.append(combiner)
        combiner = combiner_uses[0].owner.operation

    if len(combiners) != 1:
        return None, []
    return reduced, combiners


def _constant_value(value: ir.Value):
    """The numeric value of an ``arith.constant`` (scalar or splat), else None."""
    def_op = defining_op(value)
    if def_op is None or not isinstance(def_op.opview, arith.ConstantOp):
        return None
    attr = def_op.opview.value
    if isinstance(attr, (ir.FloatAttr, ir.IntegerAttr)):
        return attr.value
    if isinstance(attr, ir.DenseElementsAttr) and attr.is_splat:
        splat = attr.get_splat_value()
        return splat.value if hasattr(splat, "value") else None
    return None


def constant_int_value(value: ir.Value) -> int | None:
    """The integer value of an ``arith.constant``, or None if not constant.

    Stands in for ``getConstantIntValue``.
    """
    constant = _constant_value(value)
    return constant if isinstance(constant, int) else None


def is_defined_as_zero(value: ir.Value) -> bool:
    """Whether `value` is statically known to be zero.

    Either a constant zero scalar/splat, or chained through a ``linalg.fill`` /
    ``linalg.copy`` of a zero value. Mirrors the helper in ``FoldAddIntoDest``.
    """
    if value is None:
        return False
    constant = _constant_value(value)
    if constant is not None and constant == 0:
        return True
    def_op = defining_op(value)
    if def_op is None:
        return False
    ov = def_op.opview
    if isinstance(ov, (linalg.FillOp, linalg.CopyOp)):
        inputs = list(ov.inputs)
        return len(inputs) == 1 and is_defined_as_zero(inputs[0])
    return False


_FLOAT_WIDTHS = (
    (ir.F16Type, 16),
    (ir.BF16Type, 16),
    (ir.F32Type, 32),
    (ir.F64Type, 64),
)


def float_width(element_type: ir.Type) -> int | None:
    """Bit width of a supported float type, else None."""
    for cls, width in _FLOAT_WIDTHS:
        if isinstance(element_type, cls):
            return width
    return None


def wider_float_type(a: ir.Type, b: ir.Type) -> ir.Type | None:
    """The wider of two float types, or None if there is no common widening.

    Picks the precision a mixed-precision body is evaluated in. Equal-width types
    of different format (``f16`` vs ``bf16``) have no single-step conversion between
    them, so they are refused rather than guessed at.
    """
    width_a, width_b = float_width(a), float_width(b)
    if width_a is None or width_b is None:
        return None
    if a == b:
        return a
    if width_a == width_b:
        return None
    return a if width_a > width_b else b

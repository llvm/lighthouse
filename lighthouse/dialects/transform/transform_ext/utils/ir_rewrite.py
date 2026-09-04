"""Generic IR helpers the bindings do not provide.

Replacements for the C++ utilities the reduction fusion relies on: ``IRMapping``
plus a mapping-aware ``clone``, ``DominanceInfo``/``PostDominanceInfo``,
``getBackwardSlice``, ``AffineMap`` dim rewriting, and the zero/neutral-element
predicates from ``FoldAddIntoDest``.
"""

from mlir import ir
from mlir.dialects import arith, linalg, tensor

from lighthouse.utils.mlir import defining_op, opview

__all__ = [
    "backward_slice",
    "cast_float",
    "clone_block_body",
    "clone_op_deep_with_map",
    "clone_op_with_map",
    "constant_int_value",
    "depends_on_op",
    "float_width",
    "is_defined_as_zero",
    "match_reduction",
    "mixed_sizes",
    "move_value_definitions",
    "op_attributes",
    "operand_eliminating_constant",
    "post_dominates",
    "project_dims",
    "properly_dominates",
    "remap_dims",
    "resolve_slice_source",
    "wider_float_type",
]


# --- cloning -----------------------------------------------------------------


def op_attributes(op: ir.Operation | ir.OpView) -> dict[str, ir.Attribute]:
    """The op's discardable + inherent attributes as a name -> attr dict."""
    attrs = opview(op).operation.attributes
    return {attrs[i].name: attrs[i].attr for i in range(len(attrs))}


def clone_op_with_map(
    op: ir.Operation | ir.OpView,
    value_map: dict,
    *,
    result_type: ir.Type | None = None,
) -> ir.Operation | None:
    """Clone `op` at the current insertion point, remapping operands via `value_map`.

    Stands in for ``OpBuilder::clone(op, IRMapping)``, which the bindings do not
    expose (``Operation.clone`` takes only an insertion point). Regions are *not*
    copied, so this is limited to the region-free scalar ops that make up a
    linalg body -- which is all the fusion clones one op at a time. Results are
    recorded into `value_map`, so cloning a block in order threads the
    substitution through.

    `result_type` retypes the clone's results, which is how a body is re-emitted in
    a different precision; an ``arith.constant``'s value attribute is rebuilt to
    match. It only makes sense for the single-type scalar float ops a linalg body is
    made of -- the caller is responsible for having remapped the operands to that
    same type.
    """
    ov = opview(op)
    if any(len(r.blocks) for r in ov.operation.regions):
        return None
    attributes = op_attributes(ov)
    result_types = [r.type for r in ov.results]
    if result_type is not None:
        result_types = [result_type] * len(result_types)
        value = attributes.get("value")
        if value is not None and ir.FloatAttr.isinstance(value):
            attributes["value"] = ir.FloatAttr.get(
                result_type, ir.FloatAttr(value).value
            )
    cloned = ir.Operation.create(
        ov.operation.name,
        results=result_types,
        operands=[value_map.get(o, o) for o in ov.operands],
        attributes=attributes,
    )
    value_map.update(zip(ov.results, cloned.results))
    return cloned


def clone_op_deep_with_map(
    op: ir.Operation | ir.OpView, value_map: dict
) -> ir.Operation:
    """Deep-clone `op` (regions included) at the current insertion point.

    Operands are remapped through `value_map`, both on the op itself and on any
    value its regions capture from an enclosing scope. Results are recorded into
    `value_map`, so cloning a block's ops in order threads the substitution
    through -- which is how the reduction loop's body is copied into its
    replacement.

    Unlike `clone_op_with_map` this handles ops carrying regions (the inner
    reduction ``linalg.generic``s), at the cost of going through
    ``Operation.clone`` and patching operands afterwards.
    """
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


def move_value_definitions(
    values: list[ir.Value], before_op: ir.Operation | ir.OpView
) -> bool:
    """Move the definitions of `values` to just before `before_op`.

    Stands in for ``mlir::moveValueDefinitions``. Collects the backward slice of
    each value, keeps the ops that do not already dominate `before_op`, and moves
    them in program order so their relative order -- and therefore their
    def-before-use -- is preserved. Returns False if any op to move lives in a
    different block, which this cannot safely relocate.
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


def clone_block_body(
    src_block: ir.Block,
    arg_values: list[ir.Value],
    *,
    skip_terminator: bool = True,
    value_map: dict | None = None,
) -> dict:
    """Clone `src_block`'s ops at the current insertion point.

    `arg_values` binds the source block arguments positionally; entries may be
    None to leave an argument unbound (the caller then pre-seeds `value_map`, or
    the clone of a consuming op substitutes for it). Returns the value map, so
    the caller can look up the clone of any source value.
    """
    vmap = {} if value_map is None else value_map
    for arg, val in zip(src_block.arguments, arg_values):
        if val is not None:
            vmap[arg] = val
    ops = list(src_block.operations)
    if skip_terminator:
        ops = ops[:-1]
    for op in ops:
        clone_op_with_map(op, vmap)
    return vmap


# --- ordering / dominance ----------------------------------------------------


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
    """Whether `a` properly dominates `b`, for ops related through one block.

    The fusion legality pins the chain to a single block, so dominance reduces to
    program order there: `a` dominates `b` when the ancestor of `a` in that block
    precedes the ancestor of `b`. An op enclosing `b` dominates it. Returns False
    when no common block is found, which is the conservative answer for every
    caller here.
    """
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


# --- slices ------------------------------------------------------------------


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


def depends_on_op(value: ir.Value, op: ir.Operation | ir.OpView) -> bool:
    """Whether `value` transitively depends on `op`.

    Tells the operands that may be hoisted above the reduction loop from the ones
    computed *by* it.
    """
    target = opview(op).operation
    def_op = defining_op(value)
    if def_op is not None and def_op == target:
        return True
    return target.__hash__() in backward_slice(value)


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


# --- constants ---------------------------------------------------------------


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


#: Supported float element types with their bit widths. `f16` and `bf16` share a
#: width but not a format, so neither widens into the other.
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


def cast_float(value: ir.Value, element_type: ir.Type) -> ir.Value | None:
    """`value` converted to `element_type` via ``extf``/``truncf``, or unchanged.

    Returns None when the two types have no such conversion, which is exactly when
    `wider_float_type` refuses them.
    """
    if value.type == element_type:
        return value
    have, want = float_width(value.type), float_width(element_type)
    if have is None or want is None or have == want:
        return None
    if want > have:
        return arith.extf(element_type, value)
    return arith.truncf(element_type, value)


def operand_eliminating_constant(
    op: ir.Operation | ir.OpView, element_type: ir.Type
) -> ir.Attribute | None:
    """The constant to substitute for one operand of `op` to drop its magnitude.

    ``0`` for the additive family (``addf``/``subf``/``addi``/``subi``) and ``1`` for
    the multiplicative one (``mulf``/``divf``/``muli``) -- i.e. the neutral element of
    the family, which `arith::getNeutralElement` also gives for the commutative ops.
    Unlike that helper this answers for the non-commutative ones too, where the
    constant is *not* an identity in the left operand's position: ``0 - x`` is ``-x``
    and ``1 / x`` is the reciprocal, not ``x``.

    So the substitution generally changes the value, and the caller has to have its
    own reason why that is harmless -- for the fusion's correction term it is that the
    substituted part cancels in the new/old ratio, see `_emit_correction_term`.
    Returns None for op kinds with no such constant.
    """
    ov = opview(op)
    if isinstance(ov, (arith.AddFOp, arith.SubFOp)):
        return ir.FloatAttr.get(element_type, 0.0)
    if isinstance(ov, (arith.MulFOp, arith.DivFOp)):
        return ir.FloatAttr.get(element_type, 1.0)
    if isinstance(ov, (arith.AddIOp, arith.SubIOp)):
        return ir.IntegerAttr.get(element_type, 0)
    if isinstance(ov, arith.MulIOp):
        return ir.IntegerAttr.get(element_type, 1)
    return None


# --- affine maps -------------------------------------------------------------


def remap_dims(imap: ir.AffineMap, dim_map: dict[int, int], num_dims: int):
    """Rewrite a pure dim-projection map through `dim_map`, or None if not pure.

    Stands in for ``AffineMap::replaceDimsAndSymbols``, which the bindings do not
    expose. The fusion legality restricts every map it rewrites to a pure
    projection of plain dim exprs, so rebuilding from dim positions is exact.
    Returns None if a result is not a plain dim expr or its position is unmapped.
    """
    results = []
    for expr in imap.results:
        if not isinstance(expr, ir.AffineDimExpr):
            return None
        if expr.position not in dim_map:
            return None
        results.append(ir.AffineDimExpr.get(dim_map[expr.position]))
    return ir.AffineMap.get(num_dims, 0, results)


def project_dims(imap: ir.AffineMap, projected: set[int]):
    """Drop `projected` dims from the map's domain, renumbering the rest.

    Stands in for ``projectDims(map, dims, /*compress=*/true)``. The caller
    guarantees the map does not reference the projected dims (fusion legality
    checks exactly that), so the projection is lossless. Returns None if the map
    is not a pure dim projection or does reference a projected dim.
    """
    num_dims = imap.n_dims
    renumber: dict[int, int] = {}
    next_pos = 0
    for pos in range(num_dims):
        if pos in projected:
            continue
        renumber[pos] = next_pos
        next_pos += 1
    return remap_dims(imap, renumber, next_pos)


# --- shapes ------------------------------------------------------------------


def mixed_sizes(value: ir.Value) -> list:
    """Sizes of a shaped `value`: ints for static dims, ``tensor.dim`` otherwise.

    Stands in for ``tensor::getMixedSizes``.
    """
    shaped = ir.ShapedType(value.type)
    sizes = []
    for pos, extent in enumerate(shaped.shape):
        if ir.ShapedType.is_dynamic_size(extent):
            index = arith.constant(ir.IndexType.get(), pos)
            sizes.append(tensor.dim(value, index))
        else:
            sizes.append(extent)
    return sizes

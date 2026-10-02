"""Check whether an ``R1 -> E -> R2`` chain can share R1's tiled reduction loop.

R1 is a tiled reduction, E is an elementwise operation that consumes its running result, and R2 reduces E along
the same axis. Invalid chains raise ``FusionRejected`` with a reason.
"""

from mlir import ir
from mlir.dialects import arith, linalg, math, tensor

from lighthouse.utils.mlir import (
    defining_op,
    float_width,
    indexing_maps,
    linalg_inputs,
    linalg_outputs,
    num_loops,
    op_users,
    opview,
    reduction_dims,
)

__all__ = [
    "REDUCTION_LOOP_ATTR_NAME",
    "FusionRejected",
    "check_legal_fusion_triple",
    "collect_inner_reduction_generics",
    "collect_r1_as_elementwise_inputs",
    "find_r2_elementwise_operand",
    "map_loop_results_to_inner_reductions",
    "needs_elementwise_clone",
]

#: Marks an ``scf.for`` as a tiled reduction loop.
REDUCTION_LOOP_ATTR_NAME = "__reduction_loop__"

#: Element types the online correction is defined for.
_SUPPORTED_FLOAT_TYPES = (ir.F16Type, ir.BF16Type, ir.F32Type, ir.F64Type)


class FusionRejected(Exception):
    """The chain does not satisfy the fusion legality conditions."""


def _static_loop_ranges(op: ir.OpView) -> list[int]:
    """Recover static loop extents from operand shapes and indexing maps."""
    ranges = [ir.ShapedType.get_dynamic_size()] * num_loops(op)
    for value, imap in zip(op.operands, indexing_maps(op)):
        try:
            shape = ir.ShapedType(value.type).shape
        except (ValueError, TypeError):
            continue
        for axis, expr in enumerate(imap.results):
            if isinstance(expr, ir.AffineDimExpr):
                ranges[expr.position] = shape[axis]
    return ranges


def _resolve_slice_source(value: ir.Value) -> ir.Value:
    """Follow extract slices back to the untiled tensor."""
    while True:
        producer = defining_op(value)
        if producer is None or not isinstance(producer.opview, tensor.ExtractSliceOp):
            return value
        value = producer.opview.source


def _ancestor_in_block(op: ir.Operation | ir.OpView, block: ir.Block):
    """Find the ancestor of an op directly inside a block."""
    current = opview(op).operation
    while current is not None:
        parent = current.block
        if parent is None:
            return None
        if parent == block:
            return current
        owner = parent.owner
        current = owner.operation if owner is not None else None
    return None


def _post_dominates(a: ir.Operation | ir.OpView, b: ir.Operation | ir.OpView) -> bool:
    """Check program order for operations related through one block."""
    a_op, b_op = opview(a).operation, opview(b).operation
    if a_op == b_op:
        return True
    block = a_op.block
    if block is None:
        return False
    anchor = _ancestor_in_block(b_op, block)
    if anchor is None:
        return False
    return anchor == a_op or anchor.is_before_in_block(a_op)


def _backward_slice(value: ir.Value) -> set:
    """Collect defining operations reachable through operands."""
    found: set = set()
    producer = defining_op(value)
    if producer is None:
        return found
    stack = [producer]
    while stack:
        current = stack.pop()
        key = current.__hash__()
        if key in found:
            continue
        found.add(key)
        for operand in current.operands:
            parent = defining_op(operand)
            if parent is not None:
                stack.append(parent)
    return found


def _match_reduction(
    carried_args: list[ir.BlockArgument], position: int
) -> tuple[ir.Value | None, list]:
    """Match a single-combiner reduction in a linalg body."""
    carried = carried_args[position]
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
    block = carried.owner
    carried_set = set(carried_args)
    if reduced in carried_set:
        return None, []
    slice_ops = _backward_slice(reduced)
    if any(
        operand in carried_set
        for op in block.operations
        if op.operation.__hash__() in slice_ops
        for operand in op.operands
    ):
        return None, []

    combiners = []
    while not combiner.has_trait(ir.IsTerminatorTrait):
        if len(combiner.results) != 1 or combiner.block != block:
            return None, []
        combiner_uses = list(combiner.results[0].uses)
        if len(combiner_uses) != 1:
            return None, []
        combiners.append(combiner)
        combiner = combiner_uses[0].owner.operation
    if len(combiners) != 1:
        return None, []
    return reduced, combiners


def _constant_value(value: ir.Value):
    """Read a scalar or splat arith constant."""
    producer = defining_op(value)
    if producer is None or not isinstance(producer.opview, arith.ConstantOp):
        return None
    attr = producer.opview.value
    if isinstance(attr, (ir.FloatAttr, ir.IntegerAttr)):
        return attr.value
    if isinstance(attr, ir.DenseElementsAttr) and attr.is_splat:
        splat = attr.get_splat_value()
        return splat.value if hasattr(splat, "value") else None
    return None


def _constant_int_value(value: ir.Value) -> int | None:
    """Read an integer constant, if present."""
    constant = _constant_value(value)
    return constant if isinstance(constant, int) else None


def _is_defined_as_zero(value: ir.Value) -> bool:
    """Recognize zero constants and fills or copies of zero."""
    if value is None:
        return False
    constant = _constant_value(value)
    if constant is not None and constant == 0:
        return True
    producer = defining_op(value)
    if producer is None:
        return False
    ov = producer.opview
    if isinstance(ov, (linalg.FillOp, linalg.CopyOp)):
        inputs = list(ov.inputs)
        return len(inputs) == 1 and _is_defined_as_zero(inputs[0])
    return False


def _wider_float_type(a: ir.Type, b: ir.Type) -> ir.Type | None:
    """Find a common widening among the supported float types."""
    width_a, width_b = float_width(a), float_width(b)
    if width_a is None or width_b is None:
        return None
    if a == b:
        return a
    if width_a == width_b:
        return None
    return a if width_a > width_b else b


def collect_r1_as_elementwise_inputs(
    r1_loop: ir.OpView, e: ir.OpView
) -> tuple[list[int], list[int]]:
    """Return E inputs from the R1 loop and their loop-result indices."""
    operands: list[int] = []
    result_indices: list[int] = []
    loop_results = list(r1_loop.results)
    for operand_index, value in enumerate(linalg_inputs(e)):
        for i, result in enumerate(loop_results):
            if value == result:
                operands.append(operand_index)
                result_indices.append(i)
                break
    return operands, result_indices


def collect_inner_reduction_generics(loop: ir.OpView) -> list[ir.OpView]:
    """Return inner reduction generics in program order."""
    result = []
    for op in loop.body.operations:
        ov = opview(op)
        if isinstance(ov, linalg.GenericOp) and reduction_dims(ov):
            result.append(ov)
    return result


def map_loop_results_to_inner_reductions(loop: ir.OpView) -> list[ir.OpView | None]:
    """Map loop results to inner reductions through yielded insert slices."""
    result: list[ir.OpView | None] = [None] * len(list(loop.results))
    terminator = loop.body.operations[len(loop.body.operations) - 1]
    for idx, yielded in enumerate(terminator.operands):
        insert_op = defining_op(yielded)
        if insert_op is None or not isinstance(insert_op.opview, tensor.InsertSliceOp):
            continue
        source_op = defining_op(insert_op.opview.source)
        if source_op is None:
            continue
        ov = source_op.opview
        if isinstance(ov, linalg.GenericOp) and reduction_dims(ov):
            result[idx] = ov
    return result


def find_r2_elementwise_operand(r2: ir.OpView, e: ir.OpView) -> int:
    """Find the index of the one R2 input reading E's result."""
    found = None
    e_result = e.results[0]
    for index, value in enumerate(linalg_inputs(r2)):
        if value != e_result:
            continue
        if found is not None:
            raise FusionRejected("R2 consumes E's result more than once")
        found = index
    if found is None:
        raise FusionRejected("no R2 input is E's result")
    return found


def needs_elementwise_clone(e: ir.OpView, r2: ir.OpView) -> bool:
    """Preserve E outside the loop when another user needs its final result."""
    r2_op = r2.operation
    return any(user != r2_op for user in op_users(e.results[0]))


#: `v` is independent of every block argument (a loop-invariant scalar).
_CONST = 1 << 0
#: `v = h(x)`: no dependence on the accumulators.
_IND = 1 << 1
#: `v = alpha(m) + h(x)`: the accumulators enter additively.
_ADD = 1 << 2
#: `v = g(m) * h(x)`: the accumulators enter multiplicatively.
_MUL = 1 << 3


def _close_facts(facts: int) -> int:
    """Propagate independence to both separable forms."""
    if facts & _CONST:
        facts |= _IND
    if facts & _IND:
        facts |= _ADD | _MUL
    return facts


def check_elementwise_separability(
    e: ir.OpView, accumulator_args: list[ir.BlockArgument]
) -> None:
    """Prove E(x, m) = f(x) * g(m), so its online correction ignores x.

    A forward fact analysis tracks independence and additive or multiplicative
    separability. ``exp`` turns an additive form into a multiplicative one, as
    needed for softmax. The yielded value must depend on an accumulator.
    """
    body = e.regions[0].blocks[0]
    facts: dict = {}

    # Seed all accumulators together to prove joint separability.
    accumulators = set(accumulator_args)
    for barg in body.arguments:
        facts[barg] = _close_facts((_ADD | _MUL) if barg in accumulators else _IND)

    def facts_of(value: ir.Value) -> int:
        # Values from an enclosing scope are invariant over E's iteration space.
        return facts.get(value, _close_facts(_CONST))

    def has(value: ir.Value, fact: int) -> bool:
        return bool(facts_of(value) & fact)

    ops = list(body.operations)
    for op in ops[:-1]:
        ov = opview(op)
        # Record unknown results with no facts; absent values mean invariants.
        if len(ov.results) != 1:
            for opaque in ov.results:
                facts[opaque] = 0
            continue
        result = ov.results[0]
        f = 0

        def binary(preserved: int) -> int:
            bits = 0
            lhs, rhs = ov.operands[0], ov.operands[1]
            if has(lhs, _CONST) and has(rhs, _CONST):
                bits |= _CONST
            if has(lhs, _IND) and has(rhs, _IND):
                bits |= _IND
            if has(lhs, preserved) and has(rhs, preserved):
                bits |= preserved
            return bits

        def unary(from_fact: int, to_fact: int) -> int:
            arg = ov.operands[0]
            bits = facts_of(arg) & (_CONST | _IND)
            if has(arg, from_fact):
                bits |= to_fact
            return bits

        if isinstance(ov, arith.ConstantOp):
            f = _CONST
        elif isinstance(ov, (arith.AddFOp, arith.SubFOp)):
            f = binary(_ADD)
        elif isinstance(ov, (arith.MulFOp, arith.DivFOp)):
            f = binary(_MUL)
        elif isinstance(ov, arith.NegFOp):
            f = facts_of(ov.operands[0])
        elif isinstance(ov, (math.ExpOp, math.Exp2Op)):
            f = unary(_ADD, _MUL)
        elif isinstance(ov, (math.LogOp, math.Log2Op)):
            f = unary(_MUL, _ADD)
        elif isinstance(ov, (math.AbsFOp, math.SqrtOp, math.RsqrtOp)):
            f = unary(_MUL, _MUL)
        elif isinstance(ov, math.PowFOp):
            base, exponent = ov.operands[0], ov.operands[1]
            f = facts_of(base) & facts_of(exponent) & (_CONST | _IND)
            # A varying exponent could reintroduce data dependence.
            if has(base, _MUL) and has(exponent, _CONST):
                f |= _MUL
            if has(base, _CONST) and has(exponent, _ADD):
                f |= _MUL

        facts[result] = _close_facts(f)

    terminator = ops[-1]
    if len(terminator.operands) != 1:
        raise FusionRejected("E does not yield exactly one value")
    term = terminator.operands[0]
    if not has(term, _MUL):
        raise FusionRejected(
            "E is not multiplicatively separable in the accumulators it consumes, "
            "so no per-slice scalar can correct R2's running accumulator"
        )
    if has(term, _IND):
        raise FusionRejected("E does not depend on any consumed accumulator")


def find_elementwise_dim_for_r2_reduction_dim(
    e: ir.OpView, r2: ir.OpView, r2_e_operand: int, r2_red_dim: int
) -> int:
    """Align R2's reduction dim with E's output map for the same tensor."""
    r2_map = indexing_maps(r2)[r2_e_operand]
    e_out_map = indexing_maps(e)[len(linalg_inputs(e))]
    if len(r2_map.results) != len(e_out_map.results):
        raise FusionRejected(
            f"R2's map for E's result and E's output map have different rank "
            f"(R2: {r2_map}, E out: {e_out_map})"
        )
    for r2_expr, e_expr in zip(r2_map.results, e_out_map.results):
        if not isinstance(r2_expr, ir.AffineDimExpr) or not isinstance(
            e_expr, ir.AffineDimExpr
        ):
            raise FusionRejected(
                f"a non-dim affine expr in R2's map for E's result or in E's "
                f"output map (R2: {r2_map}, E out: {e_out_map})"
            )
        if r2_expr.position == r2_red_dim:
            return e_expr.position
    raise FusionRejected(
        f"R2's reduction dim d{r2_red_dim} does not appear in its map for E's "
        f"result: {r2_map}"
    )


def check_inner_reduction_against_elementwise(
    r1: ir.OpView, e: ir.OpView, e_tiled_dim: int, inner_results: set
) -> None:
    """Unify R1 and E loop dims through shared inputs and align reduction axes.

    Each R1 input is traced through its tile slice. The derived R1-to-E dim map
    must be complete, consistent, and map R1's reduction to ``e_tiled_dim``.
    """
    r1_red_dims = reduction_dims(r1)
    if len(r1_red_dims) != 1:
        raise FusionRejected(
            f"inner R1 does not have exactly one reduction iterator "
            f"({len(r1_red_dims)})"
        )
    if r1_red_dims[0] != num_loops(r1) - 1:
        raise FusionRejected("reduction iterator is not the innermost loop in inner R1")

    # Map R1 loop dims to E loop dims through their shared inputs.
    phi: dict[int, int] = {}

    def try_add_mapping(r1_dim: int, e_dim: int) -> bool:
        if r1_dim not in phi:
            phi[r1_dim] = e_dim
            return True
        return phi[r1_dim] == e_dim

    e_inputs = linalg_inputs(e)
    for r1_index, in1 in enumerate(linalg_inputs(r1)):
        in1_source = _resolve_slice_source(in1)
        # Sibling reduction results are running accumulators, not shared data.
        if in1_source in inner_results:
            continue
        e_index = None
        for index, candidate in enumerate(e_inputs):
            if _resolve_slice_source(candidate) == in1_source:
                e_index = index
                break
        if e_index is None:
            raise FusionRejected(f"R1 input is not also an input of E: {in1_source}")
        m1 = indexing_maps(r1)[r1_index]
        m_e = indexing_maps(e)[e_index]
        if len(m1.results) != len(m_e.results):
            raise FusionRejected(
                f"shared input has maps of different rank in R1 vs E "
                f"(R1: {m1}, E: {m_e})"
            )
        for e1, e2 in zip(m1.results, m_e.results):
            if not isinstance(e1, ir.AffineDimExpr) or not isinstance(
                e2, ir.AffineDimExpr
            ):
                raise FusionRejected(
                    f"shared input map has a non-dim affine expr (R1: {m1}, E: {m_e})"
                )
            if not try_add_mapping(e1.position, e2.position):
                raise FusionRejected(
                    f"inconsistent dim mapping between R1 and E derived from "
                    f"shared inputs (R1.d{e1.position} -> "
                    f"{{E.d{phi[e1.position]}, E.d{e2.position}}})"
                )

    if len(phi) != num_loops(r1):
        raise FusionRejected(
            f"derived dim mapping does not cover all of R1's loop dims "
            f"(covered {len(phi)} of {num_loops(r1)})"
        )
    if phi.get(r1_red_dims[0]) != e_tiled_dim:
        raise FusionRejected(
            "R1's reduction dim is not aligned with the E dim carrying R2's "
            "reduction axis under the derived dim mapping"
        )


def check_legal_fusion_triple(
    r1_loop: ir.OpView,
    result_to_inner: list[ir.OpView | None],
    e: ir.OpView,
    r2: ir.OpView,
) -> tuple[int, int]:
    """Check whether the ``R1 -> E -> R2`` chain supports online fusion.

    The checks require:

    * E and R2 each have one result/init, and E has only parallel iterators.
    * The R1 loop, E, and R2 occupy the same block.
    * Exactly one R2 input reads E's result.
    * R2 has one innermost reduction iterator. Every R2 input map contains
      that iterator and consists only of pure dimension expressions.
    * R2's map for E's result and E's output map have the same rank and only
      dimension expressions; together they identify the E reduction axis.
    * R1's loop bounds and step are constant, the step is positive, the R2
      reduction extent is static and equals the loop extent, and the step
      divides that extent. E has the same static extent on the aligned axis.
    * E consumes at least one R1 loop result. Its body depends on a consumed
      accumulator and is `multiplicatively separable` from its data inputs,
      so a per-slice correction factor exists.
    * Each R1 result read by E is broadcast over the aligned reduction axis
      through a map made only of pure dimension expressions.
    * Each consumed R1 result maps to an inner reduction (i.e. located inside tiled R1) in
      ``result_to_inner``. That reduction has one innermost reduction iterator.
      Each non-sibling input also feeds E after tracing tile slices. Their
      equal-rank, dimension-only maps yield a consistent, complete R1-to-E
      loop mapping that aligns the reduction axes.
    * R2's body is a single ``arith.addf`` reduction with a zero init (matmul contractions are allowed).
    * R2's result type is f16, bf16, f32, or f64, and E's result type can be
      widened with it to evaluate the correction.
    * Every other user of an R1 loop result post-dominates E.

    ``result_to_inner`` maps R1 loop results to the reductions producing
    their tiles. Return ``(E reduction dim, tile size)`` on success; otherwise
    raise ``FusionRejected`` with the failed condition.
    """
    if len(r2.results) != 1:
        raise FusionRejected(
            f"R2 does not have exactly one result/init ({len(r2.results)})"
        )

    if len(e.results) != 1:
        raise FusionRejected(
            f"E does not have exactly one result/init ({len(e.results)})"
        )
    if reduction_dims(e):
        raise FusionRejected(
            f"E is not all-parallel (it has {len(reduction_dims(e))} reduction loops)"
        )

    # Rewriting and the ordering checks require a shared block.
    if e.operation.block != r2.operation.block or (
        e.operation.block != r1_loop.operation.block
    ):
        raise FusionRejected("R1 loop, E and R2 are not all in the same block")

    r2_e_operand = find_r2_elementwise_operand(r2, e)

    r2_red_dims = reduction_dims(r2)
    if len(r2_red_dims) != 1:
        raise FusionRejected(
            f"R2 does not have exactly one reduction iterator ({len(r2_red_dims)})"
        )
    if r2_red_dims[0] != num_loops(r2) - 1:
        raise FusionRejected("reduction iterator is not the innermost loop in R2")

    # Each R2 input must admit slicing along the shared reduction axis.
    for index, _ in enumerate(linalg_inputs(r2)):
        imap = indexing_maps(r2)[index]
        carries_red_dim = False
        for expr in imap.results:
            if not isinstance(expr, ir.AffineDimExpr):
                raise FusionRejected(
                    f"R2 input indexing map has a non-dim affine expr: {imap}"
                )
            if expr.position == r2_red_dims[0]:
                carries_red_dim = True
        if not carries_red_dim:
            raise FusionRejected(
                f"R2 input is not reduced along R2's reduction axis (map does "
                f"not reference dim {r2_red_dims[0]}): {imap}"
            )

    e_tiled_dim = find_elementwise_dim_for_r2_reduction_dim(
        e, r2, r2_e_operand, r2_red_dims[0]
    )

    # All three extents must agree and the tile must divide the full extent.
    lb = _constant_int_value(r1_loop.lowerBound)
    ub = _constant_int_value(r1_loop.upperBound)
    step = _constant_int_value(r1_loop.step)
    if lb is None or ub is None or step is None or step <= 0:
        raise FusionRejected(
            "R1 reduction loop does not have constant, positive bounds/step"
        )
    full_extent = ub - lb
    tile_size = step
    r2_red_range = _static_loop_ranges(r2)[r2_red_dims[0]]
    if ir.ShapedType.is_dynamic_size(r2_red_range):
        raise FusionRejected(
            "R2 reduction range is dynamic; fusion requires a static reduction extent"
        )
    if r2_red_range != full_extent:
        raise FusionRejected(
            f"R2's reduction extent ({r2_red_range}) differs from the R1 loop "
            f"extent ({full_extent})"
        )
    if full_extent % tile_size != 0:
        raise FusionRejected(
            f"tile size {tile_size} does not evenly divide the reduction extent "
            f"{full_extent}"
        )

    e_red_range = _static_loop_ranges(e)[e_tiled_dim]
    if ir.ShapedType.is_dynamic_size(e_red_range) or e_red_range != full_extent:
        raise FusionRejected(
            f"E's extent along the axis carrying R2's reduction ({e_red_range}) "
            f"differs from the R1 loop extent ({full_extent})"
        )

    r1_as_e_operands, r1_result_indices = collect_r1_as_elementwise_inputs(r1_loop, e)
    if not r1_as_e_operands:
        raise FusionRejected("E does not consume any result of the R1 loop")

    accumulator_args = [
        e.regions[0].blocks[0].arguments[index] for index in r1_as_e_operands
    ]
    check_elementwise_separability(e, accumulator_args)

    inner_results = {inner.results[0] for inner in result_to_inner if inner is not None}

    # Running R1 results must be broadcast along the reduction axis.
    for index in r1_as_e_operands:
        imap = indexing_maps(e)[index]
        for expr in imap.results:
            if not isinstance(expr, ir.AffineDimExpr):
                raise FusionRejected(
                    f"R1-as-E-input indexing map has a non-dim affine expr: {imap}"
                )
            if expr.position == e_tiled_dim:
                raise FusionRejected(
                    f"R1's result is not broadcast across the E axis carrying "
                    f"R2's reduction (map references dim {expr.position}): {imap}"
                )

    for result_idx in r1_result_indices:
        inner = result_to_inner[result_idx]
        if inner is None:
            raise FusionRejected(
                f"consumed loop result {result_idx} is not produced by an inner "
                f"reduction generic"
            )
        check_inner_reduction_against_elementwise(inner, e, e_tiled_dim, inner_results)

    _, combiners = _match_reduction([r2.regions[0].blocks[0].arguments[-1]], 0)
    if not combiners:
        raise FusionRejected("R2's region does not match a reduction pattern")
    if len(combiners) != 1:
        raise FusionRejected(
            f"R2's reduction has {len(combiners)} combiners, expected exactly 1"
        )
    if not isinstance(combiners[0].opview, arith.AddFOp):
        raise FusionRejected(f"R2's combiner is not arith.addf: {combiners[0].name}")

    r2_init = linalg_outputs(r2)[0]
    if not _is_defined_as_zero(r2_init):
        raise FusionRejected("R2's init is not the additive identity (zero)")

    element_type = ir.ShapedType(r2.results[0].type).element_type
    if not isinstance(element_type, _SUPPORTED_FLOAT_TYPES):
        raise FusionRejected(
            f"R2's element type {element_type} is not a supported "
            f"floating-point type (f16/bf16/f32/f64)"
        )

    e_element_type = ir.ShapedType(e.results[0].type).element_type
    if _wider_float_type(e_element_type, element_type) is None:
        raise FusionRejected(
            f"E's element type {e_element_type} and R2's accumulator type "
            f"{element_type} have no common widening to evaluate the correction "
            f"term in"
        )

    # The replacement loop is inserted at E, so earlier R1 users would lose dominance.
    for r1_result in r1_loop.results:
        for user in op_users(r1_result):
            if user == e.operation:
                continue
            if not _post_dominates(user, e):
                raise FusionRejected(
                    f"user of an R1 result does not post-dominate E: {user.name}"
                )

    return e_tiled_dim, tile_size

"""Legality analysis for dependent-reduction (flash-attention style) fusion.

Decides whether an ``R1 -> E -> R2`` chain can be fused into ``R1``'s
already-tiled reduction loop, where

  * ``R1`` is a reduction already tiled along its reduction axis into an
    ``scf.for`` marked with ``__reduction_loop__``,
  * ``E`` is an all-parallel elementwise op consuming ``R1``'s result as a
    broadcast input plus the data inputs it shares with ``R1``,
  * ``R2`` is a reduction over ``E``'s result along the same axis.

The canonical case is two-pass softmax (``max`` then ``sum(exp)``) becoming the
online one-pass form; an attention chain is the same shape with a `P @ V`
contraction as a second ``R2``.

Rejections raise `FusionRejected` carrying the reason, which the transform op
surfaces as a silenceable failure.
"""

from mlir import ir
from mlir.dialects import arith, linalg, math, tensor

from lighthouse.utils.mlir import defining_op, op_users, opview
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.dialects.transform.transform_ext.utils import linalg_structured as ls

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

#: Unit attribute marking an ``scf.for`` as a tiled reduction loop. A plain
#: ``scf.for`` carries no iterator-type metadata, so this marker is the only way
#: to tell that the loop iterates a reduction axis. The producer of the IR
#: (e.g. ``transform.structured.tile_using_for`` followed by
#: ``transform.annotate``) is responsible for tagging it.
REDUCTION_LOOP_ATTR_NAME = "__reduction_loop__"

#: Element types the online correction is defined for.
_SUPPORTED_FLOAT_TYPES = (ir.F16Type, ir.BF16Type, ir.F32Type, ir.F64Type)


class FusionRejected(Exception):
    """The chain does not satisfy the fusion legality conditions."""


# --- chain structure ---------------------------------------------------------


def collect_r1_as_elementwise_inputs(
    r1_loop: ir.OpView, e: ir.OpView
) -> tuple[list[ls.Operand], list[int]]:
    """The `e` inputs consuming a result of the ``R1`` loop, with those result indices.

    Pre-fusion, the elementwise term consumes the loop's ``iter_arg``-carried
    results; the legality check needs both the operands (to verify their indexing
    maps) and the result index (to recover the inner reduction producing each).
    """
    operands: list[ls.Operand] = []
    result_indices: list[int] = []
    loop_results = list(r1_loop.results)
    for operand in ls.dps_input_operands(e):
        for i, result in enumerate(loop_results):
            if operand.value == result:
                operands.append(operand)
                result_indices.append(i)
                break
    return operands, result_indices


def collect_inner_reduction_generics(loop: ir.OpView) -> list[ir.OpView]:
    """The reduction ``linalg.generic``s in the loop body, in program order.

    A loop may carry several running accumulators (e.g. a fused softmax
    ``(max, sum)`` loop), each produced by its own inner reduction.
    """
    result = []
    for op in loop.body.operations:
        ov = opview(op)
        if isinstance(ov, linalg.GenericOp) and ls.num_reduction_loops(ov) != 0:
            result.append(ov)
    return result


def map_loop_results_to_inner_reductions(loop: ir.OpView) -> list[ir.OpView | None]:
    """Map each loop result to the inner reduction generic producing its tile.

    Indexed by loop result number; an entry is None when the result is not
    yielded from a reduction generic's tile (e.g. a passthrough result).

    Each running accumulator is yielded as ``tensor.insert_slice %red into %arg``,
    so each ``scf.yield`` operand is inspected: an ``insert_slice`` whose source
    is defined by a reduction generic identifies that generic.
    """
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
        if isinstance(ov, linalg.GenericOp) and ls.num_reduction_loops(ov) != 0:
            result[idx] = ov
    return result


def find_r2_elementwise_operand(r2: ir.OpView, e: ir.OpView) -> ls.Operand:
    """The single `r2` input reading `e`'s result.

    ``R2`` may have several inputs (a GEMM-like contraction does), but exactly one
    of them must be ``E``'s result.
    """
    found = None
    e_result = e.results[0]
    for operand in ls.dps_input_operands(r2):
        if operand.value != e_result:
            continue
        if found is not None:
            raise FusionRejected("R2 consumes E's result more than once")
        found = operand
    if found is None:
        raise FusionRejected("no R2 input is E's result")
    return found


def needs_elementwise_clone(e: ir.OpView, r2: ir.OpView) -> bool:
    """Whether the fusion must work on a *clone* of `e` rather than `e` itself.

    True if ``E`` has any user besides ``R2``. Fusing ``E`` computes each tile
    against the *running* ``R1`` accumulator instead of its final result, so
    ``E``'s own output is stale in every tile but the last and no user outside the
    loop may read it. The original is left in place for those users, where it still
    reads ``R1``'s final result and recomputes the term correctly.
    """
    r2_op = r2.operation
    return any(user != r2_op for user in op_users(e.results[0]))


# --- separability analysis ------------------------------------------------------------

#: `v` is independent of every block argument (a loop-invariant scalar).
_CONST = 1 << 0
#: `v = h(x)`: no dependence on the accumulators.
_IND = 1 << 1
#: `v = alpha(m) + h(x)`: the accumulators enter additively.
_ADD = 1 << 2
#: `v = g(m) * h(x)`: the accumulators enter multiplicatively.
_MUL = 1 << 3


def _close_facts(facts: int) -> int:
    """Close a fact set under the implications between the facts.

    A value the accumulators never reach trivially fits both separable shapes,
    with a trivial accumulator part (``h(x) = 0 + h(x)`` and ``h(x) = 1 * h(x)``).
    """
    if facts & _CONST:
        facts |= _IND
    if facts & _IND:
        facts |= _ADD | _MUL
    return facts


def check_elementwise_separability(
    e: ir.OpView, accumulator_args: list[ir.BlockArgument]
) -> None:
    """Verify `e` is *multiplicatively separable* in the accumulators it consumes.

    That is, that its body computes ``E(x, m) = f(x) * g(m)``, where ``m`` are the
    values read through `accumulator_args` and ``x`` is everything else. This is
    exactly the condition under which the online correction is well defined: the
    factor evaluates ``E`` twice, at the new and the old accumulator, with a
    stand-in constant for the data operands, and separability is what makes ``f``
    cancel --

        E(v, m_new) / E(v, m_old) = g(m_new) / g(m_old)  for every `v`

    so a single scalar per parallel slice repairs ``R2``'s accumulator. Without it
    the ratio depends on the stand-in and no single scalar is correct: for
    ``E = (x - m)^2`` with a tile holding ``x = 1`` and ``m`` moving from ``1`` to
    ``3``, the true sum is ``4`` while any rescale of the stale ``0`` yields ``0``.

    Implemented as an abstract interpretation over the fact sets above, in one
    forward pass -- ``E``'s body is straight-line. Accumulator block arguments are
    seeded ``_ADD | _MUL`` (a bare ``m`` is both ``m + 0`` and ``m * 1``), other
    block arguments ``_IND``, values from an enclosing scope ``_CONST``. The analysis
    is one-sided, as an abstract interpretation must be: ``_MUL`` on the yielded value
    proves separability, its absence only fails to. Accepting additionally requires
    the yield *not* to hold ``_IND``, which would mean it never reads an accumulator
    at all -- a constant-``1`` correction, i.e. not a dependent reduction.

    Tracking both shapes rather than just ``_MUL`` is what ``exp`` needs:
    ``exp(alpha(m) + h(x)) = exp(alpha(m)) * exp(h(x))`` converts an additive
    dependence into a multiplicative one, which is how softmax's ``exp(x - m)``
    reaches ``_MUL`` at all. Variance's ``(x - m)^2`` does not: ``mulf`` needs both
    operands ``_MUL`` and ``x - m`` carries only ``_ADD``.
    """
    body = e.regions[0].blocks[0]
    facts: dict = {}

    # Seed the block arguments. All tracked accumulators are seeded at once, so
    # the pass proves *joint* separability `E(x, m1, m2) = f(x) * g(m1, m2)` --
    # the correction substitutes new/old for all of them simultaneously.
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
        # Only single-result scalar arithmetic is modelled; anything else (a
        # comparison, a select, a call) is opaque, so its results are recorded with
        # no facts, which rejects the chain unless they are dead. Recorded rather
        # than skipped: an absent value is read as loop-invariant by `facts_of`,
        # which for an unmodelled result would be a wrong (and unsound) answer.
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
            # `(a1 + a2)(m) + (h1 + h2)(x)` for the additive family, and
            # `(g1 g2)(m) * (h1 h2)(x)` for the multiplicative one.
            if has(lhs, preserved) and has(rhs, preserved):
                bits |= preserved
            return bits

        def unary(from_fact: int, to_fact: int) -> int:
            # `_CONST`/`_IND` always survive a unary op.
            arg = ov.operands[0]
            bits = facts_of(arg) & (_CONST | _IND)
            if has(arg, from_fact):
                bits |= to_fact
            return bits

        if isinstance(ov, arith.ConstantOp):
            # Invariant over E's iteration space, exactly like a value from an
            # enclosing scope. Without this every value downstream of an in-body
            # constant would carry no facts, rejecting a naturally written scaled
            # term like `exp(x * cst - m)`.
            f = _CONST
        elif isinstance(ov, (arith.AddFOp, arith.SubFOp)):
            f = binary(_ADD)
        elif isinstance(ov, (arith.MulFOp, arith.DivFOp)):
            f = binary(_MUL)
        elif isinstance(ov, arith.NegFOp):
            # `-(alpha + h) = (-alpha) + (-h)` and `-(g * h) = g * (-h)`.
            f = facts_of(ov.operands[0])
        elif isinstance(ov, (math.ExpOp, math.Exp2Op)):
            f = unary(_ADD, _MUL)
        elif isinstance(ov, (math.LogOp, math.Log2Op)):
            # `log(g * h) = log(g) + log(h)`, the inverse bridge.
            f = unary(_MUL, _ADD)
        elif isinstance(ov, (math.AbsFOp, math.SqrtOp, math.RsqrtOp)):
            # `|g * h| = |g| * |h|`, and likewise for the (r)sqrt of a product.
            f = unary(_MUL, _MUL)
        elif isinstance(ov, math.PowFOp):
            base, exponent = ov.operands[0], ov.operands[1]
            f = facts_of(base) & facts_of(exponent) & (_CONST | _IND)
            # `(g * h)^p = g^p * h^p` needs a *fixed* `p`: were the exponent to
            # vary with the data, `g(m)^p(x)` would still depend on `x`.
            if has(base, _MUL) and has(exponent, _CONST):
                f |= _MUL
            # `c^(alpha + h) = c^alpha * c^h`.
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
    # A yield that never reads an accumulator would make the correction the
    # constant 1; the chain is then not a dependent reduction at all.
    if has(term, _IND):
        raise FusionRejected("E does not depend on any consumed accumulator")


# --- per-link checks ---------------------------------------------------------


def find_elementwise_dim_for_r2_reduction_dim(
    e: ir.OpView, r2: ir.OpView, r2_e_operand: ls.Operand, r2_red_dim: int
) -> int:
    """The `e` loop dim carrying `r2`'s reduction axis.

    ``E``'s output map and ``R2``'s input map both describe the same tensor
    (``E``'s result), so they align position-by-position: at tensor dim ``i``,
    ``R2``'s map yields an ``R2`` loop dim and ``E``'s output map yields an ``E``
    loop dim. The ``E`` dim sitting where ``R2`` reads its reduction dim is the
    axis along which ``E`` must be re-sliced when fused into ``R1``'s tiled loop.
    """
    r2_map = ls.indexing_map_for(r2, r2_e_operand)
    e_out_map = ls.indexing_map_for(e, ls.dps_init_operands(e)[0])
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
    """Check one inner reduction `r1` of the loop can be fused against `e`.

    ``r1`` must reduce over exactly one innermost loop, and every ``r1`` input
    (resolved *through* its tile ``extract_slice`` to the source tensor) must also
    appear as an ``e`` input.

    Fusion rewrites ``r1``'s indexing maps into ``e``'s iteration space, so the two
    ops' loop dims have to correspond. That correspondence ``phi`` is not given but
    *derived*, by unifying the maps of the inputs they share: at each position of a
    shared input's map, the ``r1`` dim indexing that tensor dim pairs with the ``e``
    dim indexing it, and the pairing must be consistent across all shared inputs.
    For softmax's ``max`` and ``exp(x - m)``, both read ``x`` as
    ``(d0, d1) -> (d0, d1)``, so ``phi = {0: 0, 1: 1}``.

    ``phi`` must additionally be

      * total over ``r1``'s loop dims, so every ``r1`` map -- its init map included
        -- can be translated, and
      * reduction-aligned: ``phi[r1_red_dim] == e_tiled_dim``, i.e. ``r1`` reduces
        the same axis that ``R2`` reduces through ``e``.

    Note the comparison is against ``E``, not ``R2``: with the elementwise term
    left unfused, ``E`` is the op sharing ``R1``'s data inputs (e.g. ``x`` in
    ``exp(x - m)``), while ``R2`` reduces ``E``'s result and need not share any
    input with ``R1``.
    """
    r1_red_dims = ls.reduction_dims(r1)
    if len(r1_red_dims) != 1:
        raise FusionRejected(
            f"inner R1 does not have exactly one reduction iterator "
            f"({len(r1_red_dims)})"
        )
    if r1_red_dims[0] != ls.num_loops(r1) - 1:
        raise FusionRejected("reduction iterator is not the innermost loop in inner R1")

    # Every input of R1 must also appear as an input of E. Unifying the two ops'
    # maps for each shared input derives phi: R1 loop dim -> E loop dim.
    phi: dict[int, int] = {}

    def try_add_mapping(r1_dim: int, e_dim: int) -> bool:
        if r1_dim not in phi:
            phi[r1_dim] = e_dim
            return True
        return phi[r1_dim] == e_dim

    e_inputs = ls.dps_input_operands(e)
    for in1 in ls.dps_input_operands(r1):
        # The inner R1 reads a `tensor.extract_slice` of the real input tensor;
        # resolve through the slice to compare against E's (untiled) inputs.
        in1_source = irr.resolve_slice_source(in1.value)
        # Inputs that are results of a sibling inner reduction (e.g. the sum
        # reduction reading the max result) are running accumulators broadcast
        # over the reduction axis; they need not appear in E, so skip them.
        if in1_source in inner_results:
            continue
        in_e = None
        for candidate in e_inputs:
            if irr.resolve_slice_source(candidate.value) == in1_source:
                in_e = candidate
                break
        if in_e is None:
            raise FusionRejected(f"R1 input is not also an input of E: {in1_source}")
        m1 = ls.indexing_map_for(r1, in1)
        m_e = ls.indexing_map_for(e, in_e)
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

    # phi must be total over R1's loop dims so R1's init map (and any other R1
    # map) can be translated into E's iteration space during fusion.
    if len(phi) != ls.num_loops(r1):
        raise FusionRejected(
            f"derived dim mapping does not cover all of R1's loop dims "
            f"(covered {len(phi)} of {ls.num_loops(r1)})"
        )
    if phi.get(r1_red_dims[0]) != e_tiled_dim:
        raise FusionRejected(
            "R1's reduction dim is not aligned with the E dim carrying R2's "
            "reduction axis under the derived dim mapping"
        )


# --- the triple --------------------------------------------------------------


def check_legal_fusion_triple(
    r1_loop: ir.OpView,
    result_to_inner: list[ir.OpView | None],
    e: ir.OpView,
    r2: ir.OpView,
) -> tuple[int, int]:
    """Verify the ``R1 -> E -> R2`` chain can be fused into `r1_loop`.

    `r1_loop` is the already-tiled producer reduction: an ``scf.for`` carrying
    running accumulators as ``iter_arg``s whose results `e` consumes.
    `result_to_inner` maps each loop result index to the inner per-tile reduction
    producing it (None for non-reduction results). The loop's ``step`` is the tile
    size and ``ub - lb`` the full reduction extent.

    Returns ``(e_tiled_dim, tile_size)``: the `e` loop dim carrying ``R2``'s
    reduction axis, which the retiling step needs, and the tile size.

    Raises `FusionRejected` with the reason on any violated condition.
    TODO: These conditions are too strict, reconsider and relax later.
    """
    # A structured op on tensors has one init per result (`num_dps_inits` is derived
    # from the result count), so one condition covers both.
    if ls.num_dps_inits(r2) != 1:
        raise FusionRejected(
            f"R2 does not have exactly one result/init ({ls.num_dps_inits(r2)})"
        )

    # (E1) E must be all-parallel with exactly one result/init: it is the
    # elementwise term feeding R2, not a reduction of its own.
    if ls.num_dps_inits(e) != 1:
        raise FusionRejected(
            f"E does not have exactly one result/init ({ls.num_dps_inits(e)})"
        )
    if ls.num_reduction_loops(e) != 0:
        raise FusionRejected(
            f"E is not all-parallel (it has {ls.num_reduction_loops(e)} "
            f"reduction loops)"
        )

    # (E2) The three ops must share a block: the rewrite builds the replacement loop
    # at E's position and reasons about program order there, and both dominance
    # helpers answer only within one block.
    if e.operation.block != r2.operation.block or (
        e.operation.block != r1_loop.operation.block
    ):
        raise FusionRejected("R1 loop, E and R2 are not all in the same block")

    # (E3) R2 may have MULTIPLE inputs (e.g. a GEMM-like contraction), but exactly
    # one must be E's result.
    r2_e_operand = find_r2_elementwise_operand(r2, e)

    # (E4) R2 must have exactly one reduction loop, and it must be its innermost.
    r2_red_dims = ls.reduction_dims(r2)
    if len(r2_red_dims) != 1:
        raise FusionRejected(
            f"R2 does not have exactly one reduction iterator ({len(r2_red_dims)})"
        )
    if r2_red_dims[0] != ls.num_loops(r2) - 1:
        raise FusionRejected("reduction iterator is not the innermost loop in R2")

    # (E5) Every R2 input must be reduced along R2's reduction axis, i.e. its
    # indexing map must reference `r2_red_dim`. An input that does *not* is
    # broadcast across the reduction (a per-parallel-slice value); such an operand
    # would have to be re-derived per tile rather than simply re-sliced, which the
    # rewrite does not model. Requiring all inputs to carry the axis is what makes
    # "re-slice every R2 input to the current tile" a complete description.
    for operand in ls.dps_input_operands(r2):
        imap = ls.indexing_map_for(r2, operand)
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

    # (E6) Locate the E loop dim carrying R2's reduction axis. This is the axis
    # along which E is re-sliced when cloned into R1's tiled loop, and the axis
    # R1's reduction dim must align with.
    e_tiled_dim = find_elementwise_dim_for_r2_reduction_dim(
        e, r2, r2_e_operand, r2_red_dims[0]
    )

    # The producer loop must have static, constant bounds: the tile size is its
    # `step` and the full reduction extent is `ub - lb`. That extent must equal
    # R2's (static) reduction extent, and the tile size must evenly divide it.
    lb = irr.constant_int_value(r1_loop.lowerBound)
    ub = irr.constant_int_value(r1_loop.upperBound)
    step = irr.constant_int_value(r1_loop.step)
    if lb is None or ub is None or step is None or step <= 0:
        raise FusionRejected(
            "R1 reduction loop does not have constant, positive bounds/step"
        )
    full_extent = ub - lb
    tile_size = step
    r2_red_range = ls.static_loop_ranges(r2)[r2_red_dims[0]]
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

    # (E7) E's extent along the axis carrying R2's reduction must match the R1
    # loop extent too, so re-slicing E to the tile is well defined.
    e_red_range = ls.static_loop_ranges(e)[e_tiled_dim]
    if ir.ShapedType.is_dynamic_size(e_red_range) or e_red_range != full_extent:
        raise FusionRejected(
            f"E's extent along the axis carrying R2's reduction ({e_red_range}) "
            f"differs from the R1 loop extent ({full_extent})"
        )

    # The E operands consuming a loop result. There must be at least one: this is
    # the R1 -> E data dependence that makes the chain fusable.
    r1_as_e_operands, r1_result_indices = collect_r1_as_elementwise_inputs(r1_loop, e)
    if not r1_as_e_operands:
        raise FusionRejected("E does not consume any result of the R1 loop")

    # (E8) E must be multiplicatively separable in the accumulators it reads, or
    # the online correction it is asked to derive does not exist.
    accumulator_args = [
        ls.matching_block_argument(e, operand) for operand in r1_as_e_operands
    ]
    check_elementwise_separability(e, accumulator_args)

    # The set of all inner reduction results in the loop, used to skip
    # sibling-reduction inputs in the per-inner-reduction phi check.
    inner_results = {inner.results[0] for inner in result_to_inner if inner is not None}

    # Each R1 result consumed by E must be broadcast across the E axis carrying
    # R2's reduction -- i.e. its indexing map must not reference `e_tiled_dim`.
    # Conservatively also require a pure dim-expr projection. This is what makes
    # the running accumulator a per-parallel-slice scalar the correction can
    # rescale.
    for operand in r1_as_e_operands:
        imap = ls.indexing_map_for(e, operand)
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

    # For each consumed loop result, recover the inner reduction producing it and
    # verify the reduction-dim alignment and shared-input phi mapping against E.
    for operand, result_idx in zip(r1_as_e_operands, r1_result_indices):
        inner = result_to_inner[result_idx]
        if inner is None:
            raise FusionRejected(
                f"consumed loop result {result_idx} is not produced by an inner "
                f"reduction generic"
            )
        check_inner_reduction_against_elementwise(inner, e, e_tiled_dim, inner_results)

    # (E9) R2's region must be a single-combiner sum reduction.
    _, combiners = irr.match_reduction(ls.region_output_args(r2), 0)
    if not combiners:
        raise FusionRejected("R2's region does not match a reduction pattern")
    if len(combiners) != 1:
        raise FusionRejected(
            f"R2's reduction has {len(combiners)} combiners, expected exactly 1"
        )
    if not isinstance(combiners[0].opview, arith.AddFOp):
        raise FusionRejected(f"R2's combiner is not arith.addf: {combiners[0].name}")

    # (E10) R2's init must be the additive identity (zero), produced directly or
    # through a linalg.fill of zero into an empty tensor.
    r2_init = ls.dps_init_operands(r2)[0].value
    if not irr.is_defined_as_zero(r2_init):
        raise FusionRejected("R2's init is not the additive identity (zero)")

    # (E11) Restrict to supported floating-point element types.
    element_type = ir.ShapedType(r2.results[0].type).element_type
    if not isinstance(element_type, _SUPPORTED_FLOAT_TYPES):
        raise FusionRejected(
            f"R2's element type {element_type} is not a supported "
            f"floating-point type (f16/bf16/f32/f64)"
        )

    # (E12) E's element type and R2's accumulator must have a common widening: the
    # correction term is E's body re-evaluated in the wider of the two, with casts on
    # the way in and out. `f16` and `bf16` have no single-step conversion between
    # them, so such a pair is rejected here rather than mid-rewrite.
    e_element_type = ir.ShapedType(e.results[0].type).element_type
    if irr.wider_float_type(e_element_type, element_type) is None:
        raise FusionRejected(
            f"E's element type {e_element_type} and R2's accumulator type "
            f"{element_type} have no common widening to evaluate the correction "
            f"term in"
        )

    # (E13) Any other user of any R1 loop result (besides E) must post-dominate E
    # so re-routing the values through the fused op is safe. Note the anchor is E,
    # not R2: E directly consumes the R1 loop results and is cloned into the loop
    # first.
    for r1_result in r1_loop.results:
        for user in op_users(r1_result):
            if user == e.operation:
                continue
            if not irr.post_dominates(user, e):
                raise FusionRejected(
                    f"user of an R1 result does not post-dominate E: {user.name}"
                )

    # Note there is no condition on the users of E's result: any user besides R2
    # makes the fusion work on a clone and leave the original E -- and therefore
    # those users -- exactly where they are. See `needs_elementwise_clone`.
    return e_tiled_dim, tile_size

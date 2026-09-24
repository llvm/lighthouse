"""Implementation for `sink_normalization_past_contraction`."""

from typing import NamedTuple

from mlir import ir
from mlir.dialects import arith, linalg

from lighthouse.utils.mlir import (
    can_cast_float,
    cast_float,
    indexing_maps,
    is_linalg_all_loops_parallel,
    linalg_inputs,
    linalg_outputs,
    num_loops,
    op_users,
    opview,
    project_dims,
    reduction_dims,
    remap_dims,
)

__all__ = ["sink_normalization_past_contraction"]

#: Body ops that factor out of a sum when their rhs is loop-invariant.
_SCALE_OPS = (arith.DivFOp, arith.MulFOp)


class _SinkPlan(NamedTuple):
    """Everything the rewrite needs, once the pair has been found legal."""

    #: Operand index of the contraction input holding the normalized value.
    operand_index: int
    #: The value the normalization scaled, which the contraction will read instead.
    numerator: ir.Value
    #: The scale itself, which moves to after the contraction.
    scale: ir.Value
    #: `numerator`'s indexing map in the contraction's loop space.
    numerator_map: ir.AffineMap
    #: `scale`'s indexing map, with the reduction dim projected away.
    scale_map: ir.AffineMap
    #: The ``divf``/``mulf`` to replicate after the contraction.
    scale_op: ir.OpView


def sink_normalization_past_contraction(normalization, contraction, rewriter):
    """Sink `normalization` past `contraction`, if that is legal.

    Returns ``(new_normalization, None)`` on success -- the `linalg.generic` now
    applying the scale to the contraction's result -- or ``(None, message)`` with a
    message explaining why the rewrite does not apply. `contraction` is rewritten in
    place, so a handle to it stays valid; `normalization` is erased.
    """
    normalization, contraction = opview(normalization), opview(contraction)
    plan, error = _plan_sink(normalization, contraction)
    if error is not None:
        return None, error
    return _apply_sink(normalization, contraction, plan, rewriter), None


def _plan_sink(normalization: ir.OpView, contraction: ir.OpView):
    """Check the pair and describe the rewrite: ``(plan, None)`` or ``(None, msg)``."""
    contraction_maps = indexing_maps(contraction)
    if contraction_maps is None:
        return None, (
            f"expected the contraction to be a structured linalg op, got "
            f"'{contraction.operation.name}'"
        )
    error = _contraction_shape_error(contraction)
    if error is not None:
        return None, f"contraction '{contraction.operation.name}': {error}"

    scale_body = _scale_body(normalization)
    if scale_body is None:
        return None, (
            f"expected the normalization to be an all-parallel two-input "
            f"linalg.generic whose body is a single arith.divf/arith.mulf of two "
            f"block arguments, got '{normalization.operation.name}'"
        )
    scale_op, num_arg, scale_arg = scale_body

    norm_maps = indexing_maps(normalization)
    out_map = norm_maps[len(linalg_inputs(normalization))]
    if not _is_identity_map(out_map):
        return (
            None,
            f"expected the normalization's output map to be the identity, got {out_map}",
        )

    users = op_users(normalization.results[0])
    if len(users) != 1:
        return None, (
            f"expected the contraction to be the normalization's only user, got "
            f"{len(users)} users"
        )

    operand_index = next(
        (
            i
            for i, value in enumerate(linalg_inputs(contraction))
            if value == normalization.results[0]
        ),
        None,
    )
    if operand_index is None:
        return None, "the normalization's result is not an input of the contraction"

    # Compose the normalization's operand maps into the contraction's loop space.
    # Its output map is the identity, so its loop dim i is the contraction's
    # `consumer_map.results[i]`.
    consumer_map = contraction_maps[operand_index]
    dim_map = {}
    for i, expr in enumerate(consumer_map.results):
        if not isinstance(expr, ir.AffineDimExpr):
            return None, (
                f"expected the contraction to read the normalization under a plain "
                f"dim projection, got {consumer_map}"
            )
        dim_map[i] = expr.position

    # Map the normalization's operand maps into the contraction's loop space.
    n_loops = num_loops(contraction)
    body_args = list(normalization.regions[0].blocks[0].arguments)
    num_index = body_args.index(num_arg)
    scale_index = body_args.index(scale_arg)
    numerator = normalization.operands[num_index]
    scale = normalization.operands[scale_index]
    num_map = remap_dims(norm_maps[num_index], dim_map, n_loops)
    scale_map = remap_dims(norm_maps[scale_index], dim_map, n_loops)
    if num_map is None or scale_map is None:
        return None, "the normalization's operand maps are not plain dim projections"

    # The factoring condition: the scale must not vary along the reduction.
    red_dim = reduction_dims(contraction)[0]
    if any(
        isinstance(r, ir.AffineDimExpr) and r.position == red_dim
        for r in scale_map.results
    ):
        return None, (
            f"the scale varies along the contraction's reduction dim d{red_dim}, so "
            f"it does not factor out of the sum"
        )

    # A named contraction's indexing maps are constrained by its own verifier, so
    # the numerator has to be readable under the map already there.
    if not isinstance(contraction, linalg.GenericOp) and num_map != consumer_map:
        return None, (
            f"'{contraction.operation.name}' cannot read the numerator under "
            f"{num_map} instead of {consumer_map}; generalize it to a "
            f"linalg.generic first"
        )

    # The contraction accumulates in its own (often wider) type, so the scale has to
    # be convertible to it -- the original scale ran on the operand's element type,
    # the sunk one runs on the accumulator's.
    accumulator_type = ir.ShapedType(contraction.results[0].type).element_type
    scale_type = ir.ShapedType(scale.type).element_type
    if not can_cast_float(scale_type, accumulator_type):
        return None, (
            f"cannot convert the scale's element type {scale_type} to the "
            f"contraction's accumulator type {accumulator_type}"
        )

    return (
        _SinkPlan(
            operand_index=operand_index,
            numerator=numerator,
            scale=scale,
            numerator_map=num_map,
            scale_map=project_dims(scale_map, {red_dim}),
            scale_op=scale_op,
        ),
        None,
    )


def _is_identity_map(imap: ir.AffineMap) -> bool:
    if imap.n_dims != len(imap.results) or imap.n_symbols != 0:
        return False
    return all(
        isinstance(r, ir.AffineDimExpr) and r.position == i
        for i, r in enumerate(imap.results)
    )


def _scale_body(op: ir.OpView):
    """``(scale_op, numerator_arg, scale_arg)`` if `op`'s body is a single scale.

    Matches a two-input all-parallel generic whose body is exactly
    ``yield numerator <div|mul> scale``, both operands being block arguments.
    """
    if not isinstance(op, linalg.GenericOp) or len(op.results) != 1:
        return None
    if not is_linalg_all_loops_parallel(op):
        return None
    if len(linalg_inputs(op)) != 2 or len(linalg_outputs(op)) != 1:
        return None
    body = op.regions[0].blocks[0]
    ops = list(body.operations)
    if len(ops) != 2:
        return None
    scale_op = opview(ops[0])
    if not isinstance(scale_op, _SCALE_OPS):
        return None
    if ops[1].operands[0] != scale_op.results[0]:
        return None
    lhs, rhs = scale_op.operands[0], scale_op.operands[1]
    args = list(body.arguments)
    if lhs not in args[:2] or rhs not in args[:2] or lhs == rhs:
        return None
    return scale_op, lhs, rhs


def _contraction_shape_error(op: ir.OpView) -> str | None:
    """Why a scale cannot be sunk past `op` on shape grounds, or None.

    Requires one reduction dim, placed last, and an identity output map. In that
    shape loop dim `i` is output dim `i` for every parallel dim, which is what lets
    the scale's indexing map be reused verbatim once the reduction dim is projected
    away.
    """
    red_dims = reduction_dims(op)
    if len(red_dims) != 1:
        return f"expected exactly one reduction dim, got {len(red_dims)}"
    n_loops = num_loops(op)
    if red_dims[0] != n_loops - 1:
        return (
            f"expected the reduction dim to be the last of {n_loops} loops, got "
            f"d{red_dims[0]}"
        )
    if len(op.results) != 1:
        return f"expected a single result, got {len(op.results)}"
    expected = ir.AffineMap.get(
        n_loops, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    actual = indexing_maps(op)[len(linalg_inputs(op))]
    if actual != expected:
        return f"expected the output map to be {expected}, got {actual}"
    return None


def _apply_sink(
    normalization: ir.OpView, contraction: ir.OpView, plan: _SinkPlan, rewriter
) -> ir.OpView:
    """Read the numerator directly, then scale the contraction's result."""
    n_loops = num_loops(contraction)

    # The contraction reads the numerator in place of the scaled operand,
    # under the numerator's map composed into the contraction's loop space. Named
    # contractions only get here when that map is the one already in place.
    maps = indexing_maps(contraction)
    contraction.operands[plan.operand_index] = plan.numerator
    if maps[plan.operand_index] != plan.numerator_map:
        maps[plan.operand_index] = plan.numerator_map
        contraction.operation.attributes["indexing_maps"] = ir.ArrayAttr.get(
            [ir.AffineMapAttr.get(m) for m in maps]
        )

    # The scale now applies once per output element, under the map the plan
    # already projected the reduction dim out of. The contraction's downstream users
    # are recorded first, so they can be rewired to the scaled result without also
    # rewiring the new op's own use of it.
    result = contraction.results[0]
    downstream = [(use.owner, use.operand_number) for use in result.uses]
    result_map = ir.AffineMap.get(
        n_loops - 1, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")

    with _insert_after(contraction), contraction.location:
        scaled = linalg.GenericOp(
            result_tensors=[result.type],
            inputs=[result, plan.scale],
            outputs=[_empty_like(result)],
            indexing_maps=ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(result_map),
                    ir.AffineMapAttr.get(plan.scale_map),
                    ir.AffineMapAttr.get(result_map),
                ]
            ),
            iterator_types=ir.ArrayAttr.get([parallel] * (n_loops - 1)),
        )
        elem = ir.ShapedType(result.type).element_type
        block = scaled.regions[0].blocks.append(
            elem, ir.ShapedType(plan.scale.type).element_type, elem
        )
        with ir.InsertionPoint(block):
            operand = cast_float(block.arguments[1], elem)
            value = type(plan.scale_op)(block.arguments[0], operand).result
            linalg.yield_([value])

    for owner, index in downstream:
        owner.operands[index] = scaled.results[0]
    rewriter.erase_op(normalization)
    return scaled


def _insert_after(op):
    """An insertion point directly after `op` in its block."""
    ops = list(op.operation.parent.regions[0].blocks[0].operations)
    idx = next(i for i, o in enumerate(ops) if o == op.operation)
    return ir.InsertionPoint(ops[idx + 1])


def _empty_like(value: ir.Value) -> ir.Value:
    from mlir.dialects import tensor

    shaped = ir.ShapedType(value.type)
    return tensor.empty(
        [shaped.get_dim_size(i) for i in range(shaped.rank)], shaped.element_type
    )

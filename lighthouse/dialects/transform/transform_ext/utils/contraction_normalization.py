"""Implementation for `sink_normalization_past_contraction`."""

from typing import NamedTuple

from mlir import ir
from mlir.dialects import arith, linalg

from lighthouse.utils.mlir import (
    can_cast_float,
    cast_float,
    clone_block_body,
    indexing_maps,
    linalg_inputs,
    linalg_outputs,
    num_loops,
    opview,
    project_dims,
    reduction_dims,
)

__all__ = ["SinkResult", "sink_normalization_past_contraction"]

#: Body ops that factor out of a sum when their rhs is loop-invariant.
_SCALE_OPS = (arith.DivFOp, arith.MulFOp)


class SinkResult(NamedTuple):
    """The ops the rewrite leaves behind."""

    #: The rebuilt contraction, without the scale operand.
    contraction: ir.OpView
    #: The `linalg.generic` applying the scale to the contraction's result.
    normalization: ir.OpView


class _Plan(NamedTuple):
    """The scale to move out, once found legal."""

    #: Operand index of the scale.
    scale_index: int
    scale: ir.Value
    #: `scale`'s indexing map, with the reduction dim projected away.
    scale_map: ir.AffineMap
    #: The body ``divf``/``mulf`` to replicate after the contraction.
    scale_op: ir.OpView


def sink_normalization_past_contraction(contraction, rewriter):
    """Sink `contraction`'s per-row scale past it, if that is legal.

    Returns ``(SinkResult, None)`` on success, or ``(None, message)`` with a message
    explaining why the rewrite does not apply.
    """
    contraction = opview(contraction)
    plan, error = _plan(contraction)
    if error is not None:
        return None, error
    return _apply(contraction, plan, rewriter), None


def _plan(contraction: ir.OpView):
    """Find the body scale to move out: ``(plan, None)`` or ``(None, message)``."""
    name = contraction.operation.name
    if not isinstance(contraction, linalg.GenericOp):
        return None, f"expected a linalg.generic, got '{name}'"
    error = _shape_error(contraction)
    if error is not None:
        return None, f"'{name}': {error}"

    body = contraction.regions[0].blocks[0]
    args = list(body.arguments)
    n_inputs = len(linalg_inputs(contraction))
    red_dim = reduction_dims(contraction)[0]
    for op in list(body.operations):
        scale_op = opview(op)
        if not isinstance(scale_op, _SCALE_OPS):
            continue
        lhs, rhs = scale_op.operands[0], scale_op.operands[1]
        # Both sides must be input block arguments, so that dropping the scale
        # leaves the numerator readable directly.
        if lhs not in args or rhs not in args:
            return None, "the body's scale does not read two block arguments"
        if args.index(lhs) >= n_inputs or args.index(rhs) >= n_inputs:
            return None, "the body's scale reads an init block argument"
        # A dead scale is not worth touching.
        if len(list(scale_op.results[0].uses)) == 0:
            continue
        scale_index = args.index(rhs)
        scale = contraction.operands[scale_index]
        scale_map = indexing_maps(contraction)[scale_index]
        error = _factors_out(contraction, scale, scale_map, red_dim)
        if error is not None:
            return None, error
        return (
            _Plan(
                scale_index=scale_index,
                scale=scale,
                scale_map=project_dims(scale_map, {red_dim}),
                scale_op=scale_op,
            ),
            None,
        )
    return None, f"'{name}' has no arith.divf/arith.mulf on two input arguments"


def _shape_error(contraction: ir.OpView) -> str | None:
    """Why no scale can be sunk past `contraction` on shape grounds, or None.

    Requires one reduction dim, placed last, and an identity output map. In that
    shape loop dim `i` is output dim `i` for every parallel dim, which lets the
    scale's indexing map be reused once the reduction dim is projected away.
    """
    red_dims = reduction_dims(contraction)
    if len(red_dims) != 1:
        return f"expected exactly one reduction dim, got {len(red_dims)}"
    n_loops = num_loops(contraction)
    if red_dims[0] != n_loops - 1:
        return (
            f"expected the reduction dim to be the last of {n_loops} loops, got "
            f"d{red_dims[0]}"
        )
    if len(contraction.results) != 1:
        return f"expected a single result, got {len(contraction.results)}"
    expected = ir.AffineMap.get(
        n_loops, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    actual = indexing_maps(contraction)[len(linalg_inputs(contraction))]
    if actual != expected:
        return f"expected the output map to be {expected}, got {actual}"
    return None


def _factors_out(
    contraction: ir.OpView, scale: ir.Value, scale_map: ir.AffineMap, red_dim: int
) -> str | None:
    """Why `scale` cannot move past the reduction, or None if it can."""
    if any(
        isinstance(r, ir.AffineDimExpr) and r.position == red_dim
        for r in scale_map.results
    ):
        return (
            f"the scale varies along the contraction's reduction dim d{red_dim}, so it "
            f"does not factor out of the sum"
        )
    # The sunk scale runs on the accumulator's element type rather than the operand's,
    # so it has to be convertible to it.
    accumulator_type = ir.ShapedType(contraction.results[0].type).element_type
    scale_type = ir.ShapedType(scale.type).element_type
    if not can_cast_float(scale_type, accumulator_type):
        return (
            f"cannot convert the scale's element type {scale_type} to the "
            f"contraction's accumulator type {accumulator_type}"
        )
    return None


def _apply(contraction: ir.OpView, plan: _Plan, rewriter) -> SinkResult:
    """Drop the in-body scale and re-apply it to the contraction's result.

    The scale operand becomes unused, so the contraction is rebuilt without it: its
    region is cloned with the matching block argument left unbound, which the clone
    of the scale's own consumer substitutes for.
    """
    body = contraction.regions[0].blocks[0]
    maps = indexing_maps(contraction)
    result = contraction.results[0]
    # The users to rewire hang off the contraction this replaces, so they are
    # recorded before anything is built.
    downstream = [(use.owner, use.operand_number) for use in result.uses]

    # The scale disappears from the body: whatever it scaled is used directly.
    numerator = plan.scale_op.operands[0]
    for use in list(plan.scale_op.results[0].uses):
        use.owner.operands[use.operand_number] = numerator
    plan.scale_op.operation.erase()

    kept_inputs = [
        v for i, v in enumerate(linalg_inputs(contraction)) if i != plan.scale_index
    ]
    kept_maps = [m for i, m in enumerate(maps) if i != plan.scale_index]
    init = linalg_outputs(contraction)[0]
    with ir.InsertionPoint(contraction), contraction.location:
        rebuilt = linalg.GenericOp(
            result_tensors=[result.type],
            inputs=kept_inputs,
            outputs=[init],
            indexing_maps=ir.ArrayAttr.get(
                [ir.AffineMapAttr.get(m) for m in kept_maps]
            ),
            iterator_types=contraction.iterator_types,
        )
        arg_types = [ir.ShapedType(v.type).element_type for v in kept_inputs]
        arg_types.append(ir.ShapedType(init.type).element_type)
        block = rebuilt.regions[0].blocks.append(*arg_types)
        with ir.InsertionPoint(block):
            new_args = iter(block.arguments)
            binding = [
                None if i == plan.scale_index else next(new_args)
                for i in range(len(list(body.arguments)))
            ]
            vmap = clone_block_body(body, binding)
            terminator = list(body.operations)[-1]
            linalg.yield_([vmap[terminator.operands[0]]])

    scaled = _emit_scale(rebuilt, plan, downstream)
    rewriter.erase_op(contraction)
    return SinkResult(contraction=rebuilt, normalization=scaled)


def _emit_scale(contraction: ir.OpView, plan: _Plan, downstream: list[tuple]):
    """Emit ``scale_op(contraction_result, scale)`` after `contraction`."""
    result = contraction.results[0]
    n_loops = num_loops(contraction)
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

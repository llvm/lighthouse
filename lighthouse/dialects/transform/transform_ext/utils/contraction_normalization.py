"""Move a per-row scale from a contraction's operand to after the contraction.

Rewrites ``contract(A / N, B)`` into ``contract(A, B) / N`` (and likewise for
``*``). Legal exactly when ``N`` is invariant along the contraction's reduction
axis, since then it factors out of the sum:

    sum_k (A[k] / N) * B[k]  ==  (sum_k A[k] * B[k]) / N

Worth doing when the reduction is long: the divide moves from one per (row, k)
element to one per (row, n) output element.

Before:

    %p = linalg.generic ins(%e, %l) { arith.divf }            (all-parallel)
    %o = linalg.generic ins(%p, %v) { mulf, addf }            (contraction over k)

After:

    %o = linalg.generic ins(%e, %v) { mulf, addf }            (contraction over k)
    %n = linalg.generic ins(%o, %l) { arith.divf }            (all-parallel)
"""

from mlir import ir
from mlir.dialects import arith, linalg

from lighthouse.utils.mlir import opview, op_users
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.dialects.transform.transform_ext.utils import linalg_structured as ls

__all__ = ["sink_normalization_past_contraction"]

#: Body ops that factor out of a sum when their rhs is loop-invariant.
_SCALE_OPS = (arith.DivFOp, arith.MulFOp)


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
    if any(it != "parallel" for it in ls.iterator_types(op)):
        return None
    if len(ls.dps_input_operands(op)) != 2 or ls.num_dps_inits(op) != 1:
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


def _standard_contraction_shape(op: ir.OpView) -> bool:
    """True for one reduction dim, placed last, with an identity output map.

    In that shape loop dim `i` is output dim `i` for every parallel dim, which is
    what lets a divisor's indexing map be reused verbatim after the reduction dim is
    projected away. Any other shape is left alone.
    """
    red_dims = ls.reduction_dims(op)
    if len(red_dims) != 1 or red_dims[0] != ls.num_loops(op) - 1:
        return False
    if len(op.results) != 1 or ls.num_dps_inits(op) != 1:
        return False
    n_loops = ls.num_loops(op)
    expected = ir.AffineMap.get(
        n_loops, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    return ls.indexing_map_for(op, ls.dps_init_operands(op)[0]) == expected


def _find_candidate(contraction: ir.OpView):
    """The input of `contraction` whose producer is a factorable scale.

    Returns ``(operand, scale_generic, numerator_value, scale_value, scale_op)`` or
    a tuple of Nones. Requires

      * the contraction to have exactly one reduction dim and an output map that is
        the identity on its parallel dims (the standard contraction shape), and
      * the scale's divisor, seen from the contraction's iteration space, not to
        reference the reduction dim -- the condition that lets it factor out.
    """
    none = (None, None, None, None, None)
    if not _standard_contraction_shape(contraction):
        return none
    red_dim = ls.reduction_dims(contraction)[0]
    n_loops = ls.num_loops(contraction)

    for operand in ls.dps_input_operands(contraction):
        producer = opview(operand.value.owner) if operand.value.owner else None
        if producer is None:
            continue
        match = _scale_body(producer)
        if match is None:
            continue
        scale_op, num_arg, scale_arg = match
        if len(op_users(producer.results[0])) != 1:
            continue
        prod_out_map = ls.indexing_map_for(producer, ls.dps_init_operands(producer)[0])
        if not _is_identity_map(prod_out_map):
            continue
        consumer_map = ls.indexing_map_for(contraction, operand)
        prod_ins = ls.dps_input_operands(producer)
        body_args = list(producer.regions[0].blocks[0].arguments)
        num_operand = prod_ins[body_args.index(num_arg)]
        scale_operand = prod_ins[body_args.index(scale_arg)]

        # Compose the producer's operand maps into the contraction's loop space.
        # The producer's output map is the identity, so its loop dim i is the
        # consumer's `consumer_map.results[i]`.
        dim_map = {}
        ok = True
        for i, expr in enumerate(consumer_map.results):
            if not isinstance(expr, ir.AffineDimExpr):
                ok = False
                break
            dim_map[i] = expr.position
        if not ok:
            continue
        num_map = irr.remap_dims(
            ls.indexing_map_for(producer, num_operand), dim_map, n_loops
        )
        scale_map = irr.remap_dims(
            ls.indexing_map_for(producer, scale_operand), dim_map, n_loops
        )
        if num_map is None or scale_map is None:
            continue
        # The factoring condition: the divisor must not vary along the reduction.
        if any(
            isinstance(r, ir.AffineDimExpr) and r.position == red_dim
            for r in scale_map.results
        ):
            continue
        return operand, producer, num_map, scale_map, scale_op
    return none


def _find_inbody_candidate(contraction: ir.OpView):
    """A scale inside `contraction`'s own body, as elementwise fusion leaves it.

    ``linalg-fuse-elementwise-ops`` sinks a producing divide into the contraction,
    giving a body like ``divf(%p, %l)`` then ``mulf``/``addf``. Returns
    ``(scale_op, scale_operand, scale_map)`` when the divisor comes straight from an
    input whose map does not reference the reduction dim, else Nones.
    """
    none = (None, None, None)
    if not _standard_contraction_shape(contraction):
        return none
    body = contraction.regions[0].blocks[0]
    args = list(body.arguments)
    inputs = ls.dps_input_operands(contraction)
    for op in list(body.operations):
        ov = opview(op)
        if not isinstance(ov, _SCALE_OPS):
            continue
        lhs, rhs = ov.operands[0], ov.operands[1]
        # The divisor must be an input block argument, not a computed value.
        if rhs not in args or args.index(rhs) >= len(inputs):
            continue
        if lhs not in args or args.index(lhs) >= len(inputs):
            continue
        scale_operand = inputs[args.index(rhs)]
        scale_map = ls.indexing_map_for(contraction, scale_operand)
        red_dim = ls.reduction_dims(contraction)[0]
        if any(
            isinstance(r, ir.AffineDimExpr) and r.position == red_dim
            for r in scale_map.results
        ):
            continue
        # The scaled value must be what feeds the accumulation, i.e. it must have a
        # use inside the body; a dead scale is not worth touching.
        if len(list(ov.results[0].uses)) == 0:
            continue
        return ov, scale_operand, scale_map
    return none


def sink_normalization_past_contraction(root: ir.Operation, rewriter) -> int:
    """Apply the rewrite to every contraction under `root`. Returns the count."""
    applied = 0
    candidates: list[ir.Operation] = []

    def visit(op: ir.Operation) -> ir.WalkResult:
        if op.name == "linalg.generic":
            candidates.append(op)
        return ir.WalkResult.ADVANCE

    root.walk(visit, ir.WalkOrder.PRE_ORDER)
    for op in candidates:
        contraction = opview(op)
        if ls.num_reduction_loops(contraction) == 0:
            continue
        operand, producer, num_map, scale_map, scale_op = _find_candidate(contraction)
        if operand is not None:
            _rewrite(
                contraction, operand, producer, num_map, scale_map, scale_op, rewriter
            )
            applied += 1
            continue
        scale_op, scale_operand, scale_map = _find_inbody_candidate(contraction)
        if scale_op is not None:
            _rewrite_inbody(contraction, scale_op, scale_operand, scale_map, rewriter)
            applied += 1
    return applied


def _rewrite_inbody(contraction, scale_op, scale_operand, scale_map, rewriter):
    """Drop the in-body scale and re-apply it to the contraction's result."""
    n_loops = ls.num_loops(contraction)
    red_dim = ls.reduction_dims(contraction)[0]
    scale = scale_operand.value
    scale_index = ls.operands_of(contraction).index(scale_operand)
    maps = [ls.indexing_map_for(contraction, o) for o in ls.operands_of(contraction)]

    # The scale disappears from the body: whatever it scaled is used directly.
    numerator = scale_op.operands[0]
    for use in list(scale_op.results[0].uses):
        use.owner.operands[use.operand_number] = numerator
    scale_op.operation.erase()

    result = contraction.results[0]
    downstream = [(use.owner, use.operand_number) for use in result.uses]
    out_scale_map = irr.project_dims(scale_map, {red_dim})

    # Rebuild the contraction without the now-unused scale operand. Its region is
    # cloned with the corresponding block argument dropped.
    kept_inputs = [
        o.value for o in ls.dps_input_operands(contraction) if o != scale_operand
    ]
    kept_maps = [m for i, m in enumerate(maps) if i != scale_index]
    init = ls.dps_init_operands(contraction)[0]
    old_args = list(contraction.regions[0].blocks[0].arguments)
    with ir.InsertionPoint(contraction), contraction.location:
        rebuilt = linalg.GenericOp(
            result_tensors=[result.type],
            inputs=kept_inputs,
            outputs=[init.value],
            indexing_maps=ir.ArrayAttr.get(
                [ir.AffineMapAttr.get(m) for m in kept_maps]
            ),
            iterator_types=contraction.iterator_types,
        )
        arg_types = [ir.ShapedType(v.type).element_type for v in kept_inputs]
        arg_types.append(ir.ShapedType(init.value.type).element_type)
        block = rebuilt.regions[0].blocks.append(*arg_types)
        with ir.InsertionPoint(block):
            binding = []
            new_iter = iter(block.arguments)
            for i, _ in enumerate(old_args):
                binding.append(None if i == scale_index else next(new_iter))
            vmap = irr.clone_block_body(contraction.regions[0].blocks[0], binding)
            terminator = list(contraction.regions[0].blocks[0].operations)[-1]
            linalg.yield_([vmap[terminator.operands[0]]])

    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    result_map = ir.AffineMap.get(
        n_loops - 1, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    with _insert_after(rebuilt), contraction.location:
        empty = _empty_like(result)
        scaled = linalg.GenericOp(
            result_tensors=[result.type],
            inputs=[rebuilt.results[0], scale],
            outputs=[empty],
            indexing_maps=ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(result_map),
                    ir.AffineMapAttr.get(out_scale_map),
                    ir.AffineMapAttr.get(result_map),
                ]
            ),
            iterator_types=ir.ArrayAttr.get([parallel] * (n_loops - 1)),
        )
        elem = ir.ShapedType(result.type).element_type
        block = scaled.regions[0].blocks.append(
            elem, ir.ShapedType(scale.type).element_type, elem
        )
        with ir.InsertionPoint(block):
            # The contraction accumulates in its own (often wider) type, so the
            # scale may need widening to match: the original scale ran on the
            # operand's element type, this one runs on the accumulator's.
            operand = irr.cast_float(block.arguments[1], elem)
            value = type(scale_op)(block.arguments[0], operand).result
            linalg.yield_([value])

    for owner, index in downstream:
        owner.operands[index] = scaled.results[0]
    rewriter.erase_op(contraction)


def _rewrite(contraction, operand, producer, num_map, scale_map, scale_op, rewriter):
    """Read the numerator directly, then scale the contraction's result."""
    n_loops = ls.num_loops(contraction)
    red_dim = ls.reduction_dims(contraction)[0]
    prod_ins = ls.dps_input_operands(producer)
    body_args = list(producer.regions[0].blocks[0].arguments)
    lhs, rhs = scale_op.operands[0], scale_op.operands[1]
    numerator = prod_ins[body_args.index(lhs)].value
    scale = prod_ins[body_args.index(rhs)].value

    # Step 1: the contraction reads the numerator in place of the scaled operand,
    # under the numerator's map composed into the contraction's loop space.
    maps = [ls.indexing_map_for(contraction, o) for o in ls.operands_of(contraction)]
    operand_index = ls.operands_of(contraction).index(operand)
    maps[operand_index] = num_map
    operand.set(numerator)
    contraction.operation.attributes["indexing_maps"] = ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(m) for m in maps]
    )

    # Step 2: the scale now applies once per output element. Its map drops the
    # reduction dim, which it provably does not reference. The contraction's
    # downstream users are recorded first, so they can be rewired to the scaled
    # result without also rewiring the scale op's own use of it.
    result = contraction.results[0]
    downstream = [(use.owner, use.operand_number) for use in result.uses]
    out_scale_map = irr.project_dims(scale_map, {red_dim})
    result_map = ir.AffineMap.get(
        n_loops - 1, 0, [ir.AffineDimExpr.get(i) for i in range(n_loops - 1)]
    )
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")

    with _insert_after(contraction), contraction.location:
        empty = _empty_like(result)
        scaled = linalg.GenericOp(
            result_tensors=[result.type],
            inputs=[result, scale],
            outputs=[empty],
            indexing_maps=ir.ArrayAttr.get(
                [
                    ir.AffineMapAttr.get(result_map),
                    ir.AffineMapAttr.get(out_scale_map),
                    ir.AffineMapAttr.get(result_map),
                ]
            ),
            iterator_types=ir.ArrayAttr.get([parallel] * (n_loops - 1)),
        )
        elem = ir.ShapedType(result.type).element_type
        block = scaled.regions[0].blocks.append(
            elem, ir.ShapedType(scale.type).element_type, elem
        )
        with ir.InsertionPoint(block):
            # The contraction accumulates in its own (often wider) type, so the
            # scale may need widening to match: the original scale ran on the
            # operand's element type, this one runs on the accumulator's.
            operand = irr.cast_float(block.arguments[1], elem)
            value = type(scale_op)(block.arguments[0], operand).result
            linalg.yield_([value])

    for owner, index in downstream:
        owner.operands[index] = scaled.results[0]
    rewriter.erase_op(producer)


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

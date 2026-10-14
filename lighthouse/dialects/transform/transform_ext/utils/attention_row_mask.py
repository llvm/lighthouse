"""Replace all-negative-infinity row masks after guarded online attention fusion."""

from mlir import ir
from mlir.dialects import arith, linalg, math, tensor

from lighthouse.utils.mlir import (
    defining_op,
    indexing_maps,
    linalg_inputs,
    linalg_outputs,
    num_loops,
    opview,
    reduction_dims,
)


def _producer(value):
    owner = defining_op(value)
    return opview(owner) if owner is not None else None


def _constant(value, expected):
    op = _producer(value)
    # A slice of a splat constant is the same splat.
    while isinstance(op, tensor.ExtractSliceOp):
        op = _producer(op.source)
    if isinstance(op, arith.ConstantOp) and isinstance(op.value, ir.DenseElementsAttr):
        attr = op.value
        return attr.is_splat and attr.get_splat_value().value == expected
    return (
        isinstance(op, arith.ConstantOp)
        and isinstance(op.value, (ir.FloatAttr, ir.IntegerAttr, ir.BoolAttr))
        and op.value.value == expected
    )


def _body_constant(consumer, value, expected):
    """A scalar constant, or a block argument fed by a splat constant input."""
    if _constant(value, expected):
        return True
    body = consumer.regions[0].blocks[0]
    inputs = list(linalg_inputs(consumer))
    arguments = list(body.arguments)[: len(inputs)]
    return value in arguments and _constant(inputs[arguments.index(value)], expected)


def _full_slice_source(value):
    while isinstance(op := _producer(value), tensor.ExtractSliceOp):
        shape = list(ir.ShapedType(op.source.type).shape)
        if (
            value.type != op.source.type
            or list(op.static_offsets) != [0] * len(shape)
            or list(op.static_sizes) != shape
            or list(op.static_strides) != [1] * len(shape)
        ):
            break
        value = op.source
    return value


def _filled_with(value, expected):
    op = _producer(_full_slice_source(value))
    if isinstance(op, tensor.ExtractSliceOp):
        return _filled_with(op.source, expected)
    return isinstance(op, linalg.FillOp) and _constant(op.inputs[0], expected)


def _negative_infinity_test(value, argument):
    compare = _producer(value)
    if not (
        isinstance(compare, arith.CmpFOp)
        and compare.predicate.value == int(arith.CmpFPredicate.OEQ)
    ):
        return False
    lhs, rhs = compare.operands
    if _constant(lhs, float("-inf")):
        lhs, rhs = rhs, lhs
    cast = _producer(lhs)
    if isinstance(cast, arith.ExtFOp):
        lhs = cast.operands[0]
    return lhs == argument and _constant(rhs, float("-inf"))


def _row_reduction(op):
    if not (
        isinstance(op, linalg.GenericOp)
        and len(linalg_inputs(op)) == 1
        and len(linalg_outputs(op)) == 1
        and reduction_dims(op) == [num_loops(op) - 1]
    ):
        return False
    rank = num_loops(op)
    maps = indexing_maps(op)
    return maps[0] == ir.AffineMap.get_identity(rank) and maps[1] == ir.AffineMap.get(
        rank, 0, [ir.AffineDimExpr.get(i) for i in range(rank - 1)]
    )


def _mask_scores(op):
    """Return the scores of an all-``-inf`` row reduction, or None.

    The ``-inf`` test is either inlined into the ``andi`` reduction or produced
    by a separate identity-mapped elementwise generic feeding it.
    """
    if not _row_reduction(op):
        return None
    body = op.regions[0].blocks[0]
    combined = _producer(body.operations[-1].operands[0])
    if not isinstance(combined, arith.AndIOp):
        return None
    lhs, rhs = combined.operands
    if lhs == body.arguments[1]:
        lhs, rhs = rhs, lhs
    if rhs != body.arguments[1] or not _filled_with(linalg_outputs(op)[0], 1):
        return None
    if _negative_infinity_test(lhs, body.arguments[0]):
        return linalg_inputs(op)[0]
    if lhs != body.arguments[0]:
        return None
    test = _producer(linalg_inputs(op)[0])
    if not (
        isinstance(test, linalg.GenericOp)
        and not reduction_dims(test)
        and len(linalg_inputs(test)) == 1
        and len(test.results) == 1
        and all(
            imap == ir.AffineMap.get_identity(num_loops(test))
            for imap in indexing_maps(test)
        )
    ):
        return None
    test_body = test.regions[0].blocks[0]
    if _negative_infinity_test(
        test_body.operations[-1].operands[0], test_body.arguments[0]
    ):
        return linalg_inputs(test)[0]
    return None


def _mask_reduction(op):
    return _mask_scores(op) is not None


def _matching_maximum(loop, mask):
    """Find a full-row running maximum over exactly this mask's score tiles."""
    scores = _mask_scores(mask)
    shape = list(ir.ShapedType(scores.type).shape)
    if not shape or any(dim <= 0 for dim in shape):
        return None
    step = _producer(loop.step)
    if not (
        _constant(loop.lowerBound, 0)
        and _constant(loop.upperBound, shape[-1])
        and isinstance(step, arith.ConstantOp)
        and isinstance(step.value, ir.IntegerAttr)
        and step.value.value > 0
        and shape[-1] % step.value.value == 0
    ):
        return None
    for index, yielded in enumerate(loop.body.operations[-1].operands):
        insert = _producer(yielded)
        if isinstance(insert, tensor.InsertSliceOp):
            if (
                insert.dest != loop.inner_iter_args[index]
                or list(insert.static_offsets) != [0] * (len(shape) - 1)
                or list(insert.static_sizes) != shape[:-1]
                or list(insert.static_strides) != [1] * (len(shape) - 1)
            ):
                continue
            yielded = insert.source
        maximum = _producer(yielded)
        if not _row_reduction(maximum):
            continue
        body = maximum.regions[0].blocks[0]
        operations = list(body.operations)
        if not (
            len(operations) == 2
            and isinstance(operations[0], arith.MaximumFOp)
            and set(operations[0].operands) == set(body.arguments)
            and operations[-1].operands[0] == operations[0].results[0]
            and _filled_with(loop.initArgs[index], float("-inf"))
            and _full_slice_source(linalg_outputs(maximum)[0])
            == loop.inner_iter_args[index]
        ):
            continue
        tile = _producer(linalg_inputs(maximum)[0])
        if (
            isinstance(tile, tensor.ExtractSliceOp)
            and tile.source == scores
            and list(tile.static_offsets)
            == [0] * (len(shape) - 1) + [ir.ShapedType.get_dynamic_size()]
            and list(tile.offsets) == [loop.induction_variable]
            and list(tile.static_sizes) == shape[:-1] + [step.value.value]
            and list(tile.static_strides) == [1] * len(shape)
            and list(ir.ShapedType(maximum.results[0].type).shape) == shape[:-1]
        ):
            return index, maximum
    return None


def _guarded_probability(value, maximum):
    """Recognize the zero-on-empty-prefix exp emitted by dependent fusion."""
    op = _producer(value)
    if not (
        isinstance(op, linalg.GenericOp)
        and not reduction_dims(op)
        and len(linalg_inputs(op)) == 2
        and len(op.results) == 1
        and linalg_inputs(op)[1] == maximum.results[0]
    ):
        return None
    score_tile = _producer(linalg_inputs(op)[0])
    max_tile = _producer(linalg_inputs(maximum)[0])
    if not (
        isinstance(score_tile, tensor.ExtractSliceOp)
        and isinstance(max_tile, tensor.ExtractSliceOp)
        and list(score_tile.operands) == list(max_tile.operands)
        and score_tile.static_offsets == max_tile.static_offsets
        and score_tile.static_sizes == max_tile.static_sizes
        and score_tile.static_strides == max_tile.static_strides
        and indexing_maps(op)
        == [
            indexing_maps(maximum)[0],
            indexing_maps(maximum)[1],
            indexing_maps(maximum)[0],
        ]
    ):
        return None
    body = op.regions[0].blocks[0]
    exp = _producer(body.operations[-1].operands[0])
    if not isinstance(exp, math.ExpOp):
        return None
    sub = _producer(exp.operands[0])
    if not isinstance(sub, arith.SubFOp) or sub.operands[0] != body.arguments[0]:
        return None
    guard = _producer(sub.operands[1])
    if (
        isinstance(guard, arith.SelectOp)
        and _negative_infinity_test(guard.operands[0], body.arguments[1])
        and _constant(guard.operands[1], 0.0)
        and guard.operands[2] == body.arguments[1]
    ):
        return op
    return None


def _prefix_mask_use(consumer, operand_index, maximum):
    """Only substitute a prefix predicate when its selected weights are zero.

    The consumer is either the PV reduction with the mask inlined or a separate
    elementwise mask producer feeding it. Either way, a prefix-empty row has
    zero guarded probabilities, so both predicates select identical values.
    """
    if not isinstance(consumer, linalg.GenericOp):
        return False
    inputs = list(linalg_inputs(consumer))
    if operand_index >= len(inputs):
        return False
    body = consumer.regions[0].blocks[0]
    argument = body.arguments[operand_index]
    uses = list(argument.uses)
    for use in uses:
        select = opview(use.owner)
        if not (
            isinstance(select, arith.SelectOp)
            and use.operand_number == 0
            and _body_constant(consumer, select.operands[1], 0.0)
            and select.operands[2] in list(body.arguments)[: len(inputs)]
        ):
            return False
        p_index = list(body.arguments).index(select.operands[2])
        probability = _guarded_probability(inputs[p_index], maximum)
        if probability is None:
            return False
        p_map = indexing_maps(consumer)[p_index]
        mask_map = indexing_maps(consumer)[operand_index]
        if tuple(mask_map.results) != tuple(p_map.results)[:-1]:
            return False
    return bool(uses)


def _compare_maximum(maximum):
    shape = ir.ShapedType(maximum.type).shape
    identity = ir.AffineMap.get_identity(len(shape))
    dest = tensor.empty(shape, ir.IntegerType.get_signless(1))
    op = linalg.GenericOp(
        result_tensors=[dest.type],
        inputs=[maximum],
        outputs=[dest],
        indexing_maps=ir.ArrayAttr.get([ir.AffineMapAttr.get(identity)] * 2),
        iterator_types=ir.ArrayAttr.get(
            [ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(shape)
        ),
    )
    body = op.regions[0].blocks.append(
        ir.ShapedType(maximum.type).element_type, ir.IntegerType.get_signless(1)
    )
    with ir.InsertionPoint(body):
        ninf = arith.constant(
            body.arguments[0].type,
            ir.FloatAttr.get(body.arguments[0].type, float("-inf")),
        )
        linalg.yield_([arith.cmpf(arith.CmpFPredicate.OEQ, body.arguments[0], ninf)])
    return op.results[0]


def rewrite_attention_row_mask(loop):
    """Rewrite supported uses, leaving unmatched masks and other users intact.

    Assumes finite or -inf scores and finite V, like guarded online attention.
    All matching is completed before mutation. Dead masks/slices are left to DCE.
    """
    plans = {}
    for consumer in loop.body.operations:
        if not isinstance(consumer, linalg.GenericOp):
            continue
        for index, value in enumerate(linalg_inputs(consumer)):
            mask = _producer(_full_slice_source(value))
            if (
                not _mask_reduction(mask)
                or mask.operation.block != loop.operation.block
            ):
                continue
            matched = _matching_maximum(loop, mask)
            if matched is None:
                continue
            result_index, maximum = matched
            if _prefix_mask_use(consumer, index, maximum):
                key = mask.operation
                if key not in plans:
                    plans[key] = (mask, result_index, maximum, [])
                plans[key][3].append((consumer, index))

    for mask, result_index, maximum, uses in plans.values():
        with ir.InsertionPoint.after(maximum), maximum.location:
            prefix = _compare_maximum(maximum.results[0])
        for consumer, index in uses:
            consumer.operands[index] = prefix
        final_uses = [
            use
            for use in mask.results[0].uses
            if use.owner.operation.block == loop.operation.block
            and loop.operation.is_before_in_block(use.owner.operation)
        ]
        if final_uses:
            with ir.InsertionPoint.after(loop), loop.location:
                final = _compare_maximum(loop.results[result_index])
            for use in final_uses:
                use.owner.operands[use.operand_number] = final

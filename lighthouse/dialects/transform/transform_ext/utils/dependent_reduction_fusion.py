"""Fuse ``R1 -> E -> R2`` into R1's tiled loop by inserting necessary online
correction for R2's running accumulator.

A new ``scf.for`` carries the old accumulators plus destinations for E and R2.
Each tile updates R1, computes E using the new R1 value, rescales the running R2
value, and reduces the tile. For example, in softmax the rescale factor for old
sum accumulator is ``exp(-m_new) / exp(-m_old)`` before adding this contribution
of current tile ``sum(exp(x_tile - m_new))``.

Recognized softmax chains use ``exp(m_old - m_new)`` and zero weights for
empty (-inf-only) tiles. It assumes finite auxiliary inputs such as V and leaves
the fully-masked-row output guard to the final normalization.

Correction factors are computed only over the dimensions their accumulator
inputs reference, then broadcast when rescaling R2. CSE can share identical
factors across consumer reductions with the same computation/accumulator types.

E's loop-carried full result is stale until the final tile and is removed after
vectorization. If E has other users, fuse a clone and leave E outside the loop
to compute its final result for them.
"""

from dataclasses import dataclass

from mlir import ir
from mlir.dialects import arith, linalg, math, scf, tensor

from lighthouse.utils.mlir import (
    indexing_maps as get_indexing_maps,
    linalg_inputs,
    linalg_outputs,
    num_loops,
    opview,
    reduction_dims,
)
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.dialects.transform.transform_ext.utils.dependent_reduction_legality import (
    FusionRejected,
    collect_r1_as_elementwise_inputs,
    find_r2_elementwise_operand,
    map_loop_results_to_inner_reductions,
)

__all__ = ["fuse_dependent_reduction_ops", "plan_correction_factor"]


def _tile_bounds(
    shaped: ir.Value,
    imap: ir.AffineMap,
    tiled_dim: int,
    iv: ir.Value,
    tile_size: int,
) -> tuple[list[int], list[ir.Value], list[int]]:
    """Return slice offsets and sizes for the current reduction tile.

    Mapped dimensions use the loop IV and tile size; other dimensions remain
    full. Dynamic offsets are returned separately for the slice builders.
    """
    shape = list(ir.ShapedType(shaped.type).shape)
    offsets: list = [0] * len(shape)
    sizes: list = list(shape)
    for pos, expr in enumerate(imap.results):
        if isinstance(expr, ir.AffineDimExpr) and expr.position == tiled_dim:
            offsets[pos] = iv
            sizes[pos] = tile_size

    static_offsets: list[int] = []
    dynamic_offsets: list[ir.Value] = []
    for offset in offsets:
        if isinstance(offset, int):
            static_offsets.append(offset)
        else:
            static_offsets.append(ir.ShapedType.get_dynamic_size())
            dynamic_offsets.append(offset)
    return static_offsets, dynamic_offsets, sizes


def _tile_slice(
    source: ir.Value,
    imap: ir.AffineMap,
    tiled_dim: int,
    iv: ir.Value,
    tile_size: int,
) -> ir.Value:
    """Extract the current tile, or the full extent for broadcast operands."""
    static_offsets, dynamic_offsets, sizes = _tile_bounds(
        source, imap, tiled_dim, iv, tile_size
    )
    result_type = ir.RankedTensorType.get(
        sizes, ir.ShapedType(source.type).element_type
    )
    return tensor.extract_slice(
        result_type,
        source,
        dynamic_offsets,
        [],
        [],
        static_offsets=static_offsets,
        static_sizes=sizes,
        static_strides=[1] * len(sizes),
    )


def _insert_tile_slice(
    tile: ir.Value,
    dest: ir.Value,
    imap: ir.AffineMap,
    tiled_dim: int,
    iv: ir.Value,
    tile_size: int,
) -> ir.Value:
    """Insert a tile into its loop-carried destination."""
    static_offsets, dynamic_offsets, sizes = _tile_bounds(
        dest, imap, tiled_dim, iv, tile_size
    )
    return tensor.insert_slice(
        tile,
        dest,
        dynamic_offsets,
        [],
        [],
        static_offsets=static_offsets,
        static_sizes=sizes,
        static_strides=[1] * len(sizes),
    )


def _clone_generic_with_operands(
    src: ir.OpView,
    inputs: list[ir.Value],
    outputs: list[ir.Value],
    result_types: list[ir.Type],
) -> ir.OpView:
    """Rebuild a generic over tiled operands, preserving maps and body."""
    op = linalg.GenericOp(
        result_tensors=result_types,
        inputs=inputs,
        outputs=outputs,
        indexing_maps=src.indexing_maps,
        iterator_types=src.iterator_types,
    )
    src_body = src.regions[0].blocks[0]
    body = op.regions[0].blocks.append(*[a.type for a in src_body.arguments])
    with ir.InsertionPoint(body):
        irr.clone_block_body(src_body, list(body.arguments), skip_terminator=False)
    return op


def _emit_elementwise(
    kind: linalg.ElementwiseKind,
    scalar_op,
    inputs: list[ir.Value],
    dest: ir.Value,
) -> ir.Value:
    """Emit ``linalg.elementwise`` and fill its region with `scalar_op`."""
    op = linalg.ElementwiseOp(
        result_tensors=[dest.type], inputs=inputs, outputs=[dest], kind=kind
    )
    element_type = ir.ShapedType(dest.type).element_type
    arg_types = [element_type] * (len(inputs) + 1)
    body = op.regions[0].blocks.append(*arg_types)
    with ir.InsertionPoint(body):
        linalg.yield_([scalar_op(body.arguments[0], body.arguments[1])])
    return op.results[0]


def _cast_tensor(value: ir.Value, element_type: ir.Type) -> ir.Value:
    """Convert a tensor elementwise to `element_type`."""
    shaped = ir.RankedTensorType(value.type)
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(shaped.rank))
    dest = tensor.empty(irr.mixed_sizes(value), element_type)
    op = linalg.GenericOp(
        result_tensors=[dest.type],
        inputs=[value],
        outputs=[dest],
        indexing_maps=ir.ArrayAttr.get([identity, identity]),
        iterator_types=ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<parallel>")
                for _ in range(shaped.rank)
            ]
        ),
    )
    body = op.regions[0].blocks.append(shaped.element_type, element_type)
    with ir.InsertionPoint(body):
        converted = irr.cast_float(body.arguments[0], element_type)
        assert converted is not None
        linalg.yield_([converted])
    return op.results[0]


@dataclass(frozen=True)
class _CorrectionPlan:
    """Read-only information needed to build both correction terms."""

    factor_maps: ir.ArrayAttr
    factor_dims: tuple[int, ...]
    compute_type: ir.Type
    neutral_by_op: tuple[ir.Attribute | None, ...]
    guard_softmax: bool = False


def _is_softmax(r1_loop: ir.OpView, e: ir.OpView) -> bool:
    indices, results = collect_r1_as_elementwise_inputs(r1_loop, e)
    body = e.regions[0].blocks[0]
    ops = list(body.operations)
    if len(indices) == 1 and len(ops) == 3:
        sub, exp, yield_op = [opview(op) for op in ops]
        inner = map_loop_results_to_inner_reductions(r1_loop)[results[0]]
        inner_ops = list(inner.regions[0].blocks[0].operations)
        if (
            isinstance(sub, arith.SubFOp)
            and isinstance(exp, math.ExpOp)
            and sub.operands[1] == body.arguments[indices[0]]
            and sub.operands[0] in list(body.arguments)[: len(linalg_inputs(e))]
            and sub.operands[0] != sub.operands[1]
            and exp.operands[0] == sub.results[0]
            and yield_op.operands[0] == exp.results[0]
            and len(inner_ops) == 2
            and isinstance(opview(inner_ops[0]), arith.MaximumFOp)
        ):
            return True
    return False


def plan_correction_factor(
    r1_loop: ir.OpView, e: ir.OpView, r2: ir.OpView
) -> _CorrectionPlan:
    """Assemble correction maps and constants after fusion legality succeeds."""
    guard_softmax = _is_softmax(r1_loop, e)
    accumulator_indices, _ = collect_r1_as_elementwise_inputs(r1_loop, e)
    r2_e_index = find_r2_elementwise_operand(r2, e)
    e_out_map = get_indexing_maps(e)[len(linalg_inputs(e))]
    r2_e_map = get_indexing_maps(r2)[r2_e_index]
    # The legality check established equal-rank, dimension-only maps.
    e_dim_to_r2 = {
        e_dim.position: r2_dim.position
        for e_dim, r2_dim in zip(e_out_map.results, r2_e_map.results)
    }
    r2_red_dim = reduction_dims(r2)[0]
    r2_type = ir.RankedTensorType(r2.results[0].type)
    rank = r2_type.rank
    factor_maps = []
    for index in accumulator_indices:
        # Legality requires pure E input maps broadcast along R2's reduction
        # and an identity R2 output map over the parallel dimensions.
        mapped = irr.remap_dims(get_indexing_maps(e)[index], e_dim_to_r2, num_loops(r2))
        assert mapped is not None
        projected = irr.project_dims(mapped, {r2_red_dim})
        assert projected is not None and projected.n_dims == rank
        factor_maps.append(projected)
    factor_dims = tuple(
        sorted({expr.position for imap in factor_maps for expr in imap.results})
    )
    unused_dims = set(range(rank)) - set(factor_dims)
    compact_maps = []
    for imap in factor_maps:
        compact = irr.project_dims(imap, unused_dims)
        assert compact is not None
        compact_maps.append(compact)
    compact_maps.append(ir.AffineMap.get_identity(len(factor_dims)))

    e_type = ir.ShapedType(e.results[0].type).element_type
    compute_type = irr.wider_float_type(e_type, r2_type.element_type)
    assert compute_type is not None  # Checked by check_legal_fusion_triple.

    body = e.regions[0].blocks[0]
    accumulator_args = {body.arguments[index] for index in accumulator_indices}
    neutrals = []
    for operation in list(body.operations)[:-1]:
        op = opview(operation)
        data_args = [
            operand
            for operand in op.operands
            if isinstance(operand, ir.BlockArgument)
            and operand.owner == body
            and operand not in accumulator_args
        ]
        # TODO: This assumes all the ops in E's body is data-dependent on the R1 results.
        # This is not true all the time, e.g., (x + x) * exp(x - m). Op (x + x)
        # does not depend on the R1 results, and therefore will be neutralized to
        # zero. Needs more analysis on E's body to correctly handle such cases.
        neutral = (
            irr.operand_eliminating_constant(op, compute_type) if data_args else None
        )
        neutrals.append(neutral)
    return _CorrectionPlan(
        ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in compact_maps]),
        factor_dims,
        compute_type,
        tuple(neutrals),
        guard_softmax,
    )


def _emit_correction_term(
    e: ir.OpView,
    bindings: list[tuple[int, ir.Value]],
    plan: _CorrectionPlan,
) -> ir.Value:
    """Evaluate E's body at `compute_type` for one accumulator state.

    `bindings` maps E input indices to new or old accumulator values. Unbound
    data inputs get 0 for additive ops or 1 for multiplicative ops; separability
    makes their factor cancel in ``term(new) / term(old)``. Captured values are
    cast to the plan's compute type. The plan checks all failure conditions.

    TODO: A stand-in that makes the data factor zero yields ``0 / 0``; legality
    currently does not reject that case.
    """
    body = e.regions[0].blocks[0]
    compute_type = plan.compute_type
    ops = list(body.operations)
    term = ops[-1].operands[0]

    # Bind the accumulator block arguments; data arguments stay unmapped and are
    # neutralized per consuming op below.
    value_map: dict = {}
    for index, value in bindings:
        value_map[body.arguments[index]] = value

    for op_index, op in enumerate(ops[:-1]):
        ov = opview(op)
        temporary: list = []
        for operand in ov.operands:
            if operand in value_map:
                continue
            if isinstance(operand, ir.BlockArgument) and operand.owner == body:
                neutral = plan.neutral_by_op[op_index]
                assert neutral is not None
                value_map[operand] = arith.constant(compute_type, neutral)
                temporary.append(operand)
            else:
                # Captured from an enclosing scope: kept, but at `compute_type`. Not
                # dropped afterwards -- unlike a neutral element it is reusable.
                converted = irr.cast_float(operand, compute_type)
                assert converted is not None
                value_map[operand] = converted
        irr.clone_op_with_map(ov, value_map, result_type=compute_type)
        # Drop the per-op substitutions so the next consumer of the same data
        # argument gets its own neutral element.
        for operand in temporary:
            del value_map[operand]

    return value_map.get(term, term)


def _correction_factor(
    e: ir.OpView,
    r2: ir.OpView,
    accumulators: list[tuple[int, ir.Value, ir.Value]],
    r2_accumulator: ir.Value,
    plan: _CorrectionPlan,
) -> ir.Value:
    """Build ``term(new) / term(old)`` only over its dependent dimensions.

    Use the prevalidated maps and compute type, then narrow the ratio to R2's
    accumulator type if needed.
    """
    accumulator_type = ir.RankedTensorType(r2_accumulator.type)
    rank = len(plan.factor_dims)
    iterator_types = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )
    element_type = accumulator_type.element_type
    compute_type = plan.compute_type
    accumulator_sizes = irr.mixed_sizes(r2_accumulator)
    init = tensor.empty(
        [accumulator_sizes[dim] for dim in plan.factor_dims], compute_type
    )

    if plan.guard_softmax:
        _, new_max, old_max = accumulators[0]
        op = linalg.GenericOp(
            result_tensors=[init.type],
            inputs=[new_max, old_max],
            outputs=[init],
            indexing_maps=ir.ArrayAttr.get(
                [plan.factor_maps[0], plan.factor_maps[0], plan.factor_maps[1]]
            ),
            iterator_types=iterator_types,
        )
        body = op.regions[0].blocks.append(
            ir.ShapedType(new_max.type).element_type,
            ir.ShapedType(old_max.type).element_type,
            compute_type,
        )
        with ir.InsertionPoint(body):
            new = irr.cast_float(body.arguments[0], compute_type)
            old = irr.cast_float(body.arguments[1], compute_type)
            zero = arith.constant(compute_type, ir.FloatAttr.get(compute_type, 0.0))
            ninf = arith.constant(
                compute_type, ir.FloatAttr.get(compute_type, float("-inf"))
            )
            empty = arith.cmpf(arith.CmpFPredicate.OEQ, old, ninf)
            factor = math.exp(arith.subf(old, new))
            linalg.yield_([arith.select(empty, zero, factor)])
        factor = op.results[0]
        return (
            factor
            if compute_type == element_type
            else _cast_tensor(factor, element_type)
        )

    def build_term(pick) -> ir.Value:
        inputs = [pick(acc) for acc in accumulators]
        op = linalg.GenericOp(
            result_tensors=[init.type],
            inputs=inputs,
            outputs=[init],
            indexing_maps=plan.factor_maps,
            iterator_types=iterator_types,
        )
        # A block argument type follows its operand, so the accumulators enter the
        # body at their own element type and are converted inside it.
        arg_types = [ir.ShapedType(v.type).element_type for v in inputs]
        body = op.regions[0].blocks.append(*arg_types, compute_type)
        with ir.InsertionPoint(body):
            bindings = []
            for (operand, _, _), arg in zip(accumulators, list(body.arguments)[:-1]):
                converted = irr.cast_float(arg, compute_type)
                assert converted is not None
                bindings.append((operand, converted))
            term = _emit_correction_term(e, bindings, plan)
            # `E` may yield a bare accumulator or a captured value; either way the
            # yielded type has to be the output's.
            converted_term = irr.cast_float(term, compute_type)
            assert converted_term is not None
            linalg.yield_([converted_term])
        return op.results[0]

    term_new = build_term(lambda acc: acc[1])
    term_old = build_term(lambda acc: acc[2])

    # TODO: Evaluating two exp() values and dividing them may lead to numerical instability.
    # Rewriting the division into substraction require `reassoc` flags on exp. Explicitily check
    # these flags before doing the transformation.
    factor = _emit_elementwise(
        linalg.ElementwiseKind.div,
        lambda a, b: arith.divf(a, b),
        [term_new, term_old],
        init,
    )
    if compute_type == element_type:
        return factor
    # Narrow the ratio back to R2's accumulator type, which the rescale multiply and
    # the fused R2 both work in.
    return _cast_tensor(factor, element_type)


def _rescale_accumulator(
    accumulator: ir.Value, factor: ir.Value, factor_dims: tuple[int, ...]
) -> ir.Value:
    """Broadcast the compact factor directly into the accumulator multiply."""
    shaped = ir.RankedTensorType(accumulator.type)
    if len(factor_dims) == shaped.rank:
        return _emit_elementwise(
            linalg.ElementwiseKind.mul,
            lambda a, b: arith.mulf(a, b),
            [accumulator, factor],
            accumulator,
        )
    identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(shaped.rank))
    broadcast = ir.AffineMapAttr.get(
        ir.AffineMap.get(
            shaped.rank, 0, [ir.AffineDimExpr.get(dim) for dim in factor_dims]
        )
    )
    op = linalg.GenericOp(
        result_tensors=[accumulator.type],
        inputs=[accumulator, factor],
        outputs=[accumulator],
        indexing_maps=ir.ArrayAttr.get([identity, broadcast, identity]),
        iterator_types=ir.ArrayAttr.get(
            [
                ir.Attribute.parse("#linalg.iterator_type<parallel>")
                for _ in range(shaped.rank)
            ]
        ),
    )
    body = op.regions[0].blocks.append(*[shaped.element_type] * 3)
    with ir.InsertionPoint(body):
        linalg.yield_([arith.mulf(body.arguments[0], body.arguments[1])])
    return op.results[0]


def _tiled_dim_of_input(
    consumer: ir.OpView, operand_index: int, consumer_dim: int
) -> int | None:
    """Return the producer tensor dim that `consumer_dim` reads through an input."""
    maps = get_indexing_maps(consumer)
    for position, expr in enumerate(maps[operand_index].results):
        if isinstance(expr, ir.AffineDimExpr) and expr.position == consumer_dim:
            return position
    return None


def _chain_tiled_dims(
    e: ir.OpView,
    chain: list[ir.OpView],
    r2: ir.OpView,
    r2_red_dim: int,
    e_tiled_dim: int,
) -> list[int]:
    """Return each intermediate's loop dim that follows R2's reduction tile."""
    tiled_dims = []
    for op in chain:
        if reduction_dims(op) or len(op.results) != 1:
            raise FusionRejected("intermediates must be single-result elementwise ops")
    consumer, consumer_dim = r2, r2_red_dim
    for producer in reversed([e, *chain]):
        indices = [
            index
            for index, operand in enumerate(linalg_inputs(consumer))
            if operand == producer.results[0]
        ]
        if not indices:
            raise FusionRejected("intermediate chain is not connected to R2")
        position = _tiled_dim_of_input(consumer, indices[0], consumer_dim)
        if position is None:
            raise FusionRejected("intermediate does not carry R2's reduction axis")
        out_map = get_indexing_maps(producer)[len(linalg_inputs(producer))]
        expr = out_map.results[position]
        if not isinstance(expr, ir.AffineDimExpr):
            raise FusionRejected("intermediate output map is not a projection")
        consumer, consumer_dim = producer, expr.position
        tiled_dims.insert(0, consumer_dim)
    if tiled_dims[0] != e_tiled_dim:
        raise FusionRejected("intermediate chain changes E's tiled axis")
    return tiled_dims[1:]


def _shift_index_ops(op: ir.OpView, dim: int, offset: ir.Value) -> None:
    """Offset ``linalg.index`` of a tiled dim by the tile start."""
    for inner in list(op.regions[0].blocks[0].operations):
        if isinstance(inner, linalg.IndexOp) and inner.dim.value == dim:
            uses = list(inner.result.uses)
            with ir.InsertionPoint.after(inner):
                shifted = arith.addi(inner.result, offset)
            for use in uses:
                use.owner.operands[use.operand_number] = shifted


def fuse_dependent_reduction_ops(
    rewriter,
    r1_loop: ir.OpView,
    e: ir.OpView,
    r2: ir.OpView,
    e_tiled_dim: int,
    tile_size: int,
    correction_plan: _CorrectionPlan,
    chain: list[ir.OpView] | None = None,
) -> ir.OpView:
    """Rebuild R1's loop with fused E and R2, using the checked tile axis.

    `chain` lists all-parallel intermediates between E and R2. They are tiled
    as separate ops inside the loop, so R2 keeps its own body. The in-loop E
    and chain tiles use the running R1 value and only feed R2, so their full
    extent is not carried; the originals keep serving other users with the
    final R1 value and are erased once dead.
    """
    r1_loop, e, r2 = opview(r1_loop), opview(e), opview(r2)
    chain = [opview(op) for op in (chain or [])]
    r2_red_dim = reduction_dims(r2)[0]
    num_old_results = len(list(r1_loop.results))
    chain_dims = (
        _chain_tiled_dims(e, chain, r2, r2_red_dim, e_tiled_dim) if chain else []
    )

    # This can reject cross-block moves; do it before cloning E or changing R2.
    internal = {op.results[0] for op in (e, *chain)}
    to_hoist = [
        operand
        for consumer in (e, *chain, r2)
        for operand in consumer.operands
        if operand not in internal and not irr.depends_on_op(operand, r1_loop)
    ]
    if not irr.move_value_definitions(to_hoist, e):
        raise FusionRejected(
            "could not move E and R2's operand definitions above the reduction loop"
        )

    # Build at E's position, after its init and before later R1 users.
    anchor = e
    e_dest = linalg_outputs(e)[0]
    r2_dest = linalg_outputs(r2)[0]

    result_to_inner = map_loop_results_to_inner_reductions(r1_loop)
    accumulator_indices, accumulator_result_indices = collect_r1_as_elementwise_inputs(
        r1_loop, e
    )

    # Build the new fused loop.
    with ir.InsertionPoint(anchor), r1_loop.location:
        new_loop = scf.ForOp(
            r1_loop.lowerBound,
            r1_loop.upperBound,
            r1_loop.step,
            list(r1_loop.initArgs) + [r2_dest],
        )
    # Preserve the reduction marker for a later fusion.
    for name, attr in irr.op_attributes(r1_loop).items():
        new_loop.operation.attributes[name] = attr

    iv = new_loop.induction_variable
    r2_result_index = num_old_results
    r2_arg = new_loop.inner_iter_args[r2_result_index]

    value_map: dict = {r1_loop.induction_variable: iv}
    value_map.update(zip(r1_loop.inner_iter_args, new_loop.inner_iter_args))

    old_body_ops = list(r1_loop.body.operations)
    with ir.InsertionPoint(new_loop.body), r1_loop.location:
        # Clone the tiled R1 body over the new loop arguments.
        for op in old_body_ops[:-1]:
            irr.clone_op_deep_with_map(op, value_map)

        # --- Construct fused E, re-sliced to the current tile ---
        # Online correction needs R1's value before and after current tile. Use the
        # inner reduction's init as the old value: it is the exact tile the reduction
        # reads, so we need not reconstruct that tile from the loop argument.
        accumulator_values = {}
        for index, result_idx in zip(accumulator_indices, accumulator_result_indices):
            inner = result_to_inner[result_idx]
            new_value = value_map[inner.results[0]]
            old_init = linalg_outputs(inner)[0]
            old_value = value_map.get(old_init, old_init)
            accumulator_values[index] = (new_value, old_value)

        fused_e_inputs = []
        for index, operand in enumerate(linalg_inputs(e)):
            if index in accumulator_values:
                # Accumulators broadcast across the reduction axis.
                fused_e_inputs.append(accumulator_values[index][0])
            else:
                # Otherwise slice the operand to the current tile.
                fused_e_inputs.append(
                    _tile_slice(
                        operand,
                        get_indexing_maps(e)[index],
                        e_tiled_dim,
                        iv,
                        tile_size,
                    )
                )
        e_out_map = get_indexing_maps(e)[len(linalg_inputs(e))]
        # A fresh tile: equal E tiles of sibling fusions can then be CSE'd.
        e_dest_tile = _tile_slice(e_dest, e_out_map, e_tiled_dim, iv, tile_size)
        fused_e = _clone_generic_with_operands(
            e, fused_e_inputs, [e_dest_tile], [e_dest_tile.type]
        )
        if correction_plan.guard_softmax:
            sub = opview(fused_e.regions[0].blocks[0].operations[0])
            maximum = sub.operands[1]
            with ir.InsertionPoint(sub):
                zero = arith.constant(maximum.type, ir.FloatAttr.get(maximum.type, 0.0))
                ninf = arith.constant(
                    maximum.type, ir.FloatAttr.get(maximum.type, float("-inf"))
                )
                empty = arith.cmpf(arith.CmpFPredicate.OEQ, maximum, ninf)
                # exp(-inf - 0) is zero, unlike exp(-inf - -inf).
                sub.operands[1] = arith.select(empty, zero, maximum)

        # --- Construct the online correction that rescales R2's running accumulator ---
        r2_out_map = get_indexing_maps(r2)[len(linalg_inputs(r2))]
        r2_acc_tile = _tile_slice(r2_arg, r2_out_map, r2_red_dim, iv, tile_size)
        accumulators = [
            (index, new_value, old_value)
            for index, (new_value, old_value) in accumulator_values.items()
        ]
        factor = _correction_factor(e, r2, accumulators, r2_acc_tile, correction_plan)
        corrected_r2_acc = _rescale_accumulator(
            r2_acc_tile, factor, correction_plan.factor_dims
        )

        # --- Tile the intermediates as separate ops between E and R2 ---
        tiled_values = {e.results[0]: fused_e.results[0]}
        for op, tiled_dim in zip(chain, chain_dims):
            maps = get_indexing_maps(op)
            num_inputs = len(linalg_inputs(op))
            inputs = [
                tiled_values[operand]
                if operand in tiled_values
                else _tile_slice(operand, maps[index], tiled_dim, iv, tile_size)
                for index, operand in enumerate(linalg_inputs(op))
            ]
            dest = _tile_slice(
                linalg_outputs(op)[0], maps[num_inputs], tiled_dim, iv, tile_size
            )
            tiled = _clone_generic_with_operands(op, inputs, [dest], [dest.type])
            _shift_index_ops(tiled, tiled_dim, iv)
            tiled_values[op.results[0]] = tiled.results[0]

        # --- Build fused R2 that accumulates this tile into the rescaled sum ---
        r2_inputs = []
        for index, operand in enumerate(linalg_inputs(r2)):
            if operand in tiled_values:
                r2_inputs.append(tiled_values[operand])
            else:
                r2_inputs.append(
                    _tile_slice(
                        operand,
                        get_indexing_maps(r2)[index],
                        r2_red_dim,
                        iv,
                        tile_size,
                    )
                )
        fused_r2 = _clone_generic_with_operands(
            r2, r2_inputs, [corrected_r2_acc], [r2.results[0].type]
        )
        _shift_index_ops(fused_r2, r2_red_dim, iv)

        # --- the new yield ---
        yielded = [value_map[o] for o in old_body_ops[-1].operands]
        yielded.append(
            _insert_tile_slice(
                fused_r2.results[0], r2_arg, r2_out_map, r2_red_dim, iv, tile_size
            )
        )
        scf.YieldOp(yielded)

    # --- retire the originals -------------------------------------------------
    rewriter.replace_op(r2, [new_loop.results[r2_result_index]])
    for op in reversed([e, *chain]):
        if not list(op.results[0].uses):
            rewriter.erase_op(op)
    rewriter.replace_op(r1_loop, list(new_loop.results)[:num_old_results])
    return new_loop

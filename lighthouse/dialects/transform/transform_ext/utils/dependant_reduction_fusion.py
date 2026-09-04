"""The dependent-reduction (flash-attention style) fusion rewrite.

Fuses an ``R1 -> E -> R2`` chain into ``R1``'s already-tiled reduction loop,
turning a two-pass reduction into an online one-pass loop. See
`dependant_reduction_legality` for the chain's shape and the conditions checked
before this runs.

The rewrite *rebuilds* the loop rather than mutating it: MLIR cannot grow an
``scf.for``'s ``iter_args`` in place, so a new loop is created carrying the
original accumulators plus one for ``E``'s result and one for ``R2``'s, the
original body is cloned into it, and the fused ops are appended.

For softmax ``m = max_j x``, ``p = exp(x - m)``, ``s = sum_j p`` the result is::

    %loop:3 = scf.for %iv = 0 to 512 step 32
        iter_args(%mArg = %mInit, %eArg = %eInit, %sArg = %sInit) {
      %xt   = tensor.extract_slice %X[0, %iv] [64, 32] [1, 1]
      %mOld = tensor.extract_slice %mArg[0] [64] [1]
      %mNew = linalg.generic ins(%xt) outs(%mOld) { arith.maximumf }   // R1
      %et   = tensor.extract_slice %eArg[0, %iv] [64, 32] [1, 1]
      %p    = linalg.generic ins(%xt, %mNew) outs(%et) { subf; exp }   // fused E
      %sOld = tensor.extract_slice %sArg[0] [64] [1]
      %tNew = linalg.generic ins(%mNew) outs(...) { subf 0, m; exp }   // correction
      %tOld = linalg.generic ins(%mOld) outs(...) { subf 0, m; exp }
      %f    = linalg.elementwise kind=div ins(%tNew, %tOld)
      %sc   = linalg.elementwise kind=mul ins(%sOld, %f)
      %sNew = linalg.generic ins(%p) outs(%sc) { arith.addf }          // fused R2
      scf.yield %mNew, insert_slice(%p into %eArg), insert_slice(%sNew into %sArg)
    }

``%eArg`` carries ``E``'s full-extent result only so the fused op has a destination;
every tile but the last is stale (computed against the *running* accumulator), so
nothing may read it off the loop. It is dead on arrival, but not provably so until
vectorization has replaced the in-loop destination slice with transfer ops -- so the
schedule runs ``remove-dead-values`` after vectorizing to unwind it.

The op fused is ``E`` itself, unless ``E`` has a consumer besides ``R2``: then a
clone is fused and the original stays put for that consumer, still reading ``R1``'s
final result. See ``needs_elementwise_clone``.
"""

from mlir import ir
from mlir.dialects import arith, linalg, scf, tensor

from lighthouse.utils.mlir import opview
from lighthouse.dialects.transform.transform_ext.utils import ir_rewrite as irr
from lighthouse.dialects.transform.transform_ext.utils import linalg_structured as ls
from lighthouse.dialects.transform.transform_ext.utils.dependant_reduction_legality import (
    FusionRejected,
    collect_r1_as_elementwise_inputs,
    find_r2_elementwise_operand,
    map_loop_results_to_inner_reductions,
    needs_elementwise_clone,
)

__all__ = ["fuse_dependant_reduction_ops"]


def _tile_bounds(
    shaped: ir.Value,
    imap: ir.AffineMap,
    tiled_dim: int,
    iv: ir.Value,
    tile_size: int,
) -> tuple[list[int], list[ir.Value], list[int]]:
    """Offsets/sizes selecting the current reduction tile of `shaped`.

    Every tensor position whose map result references `tiled_dim` is cut to the
    tile (offset = the loop IV, size = `tile_size`); the rest span their full
    extent. Returns the offsets split into the static list (with
    ``kDynamic`` where the IV goes) and the matching dynamic values, plus the
    sizes -- the form the slice builders take.
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
    """Extract the current reduction tile of `source`, per its indexing map.

    This is what re-slices a fused op's full-extent operands down to the tile the
    enclosing loop is on. An operand not spanning `tiled_dim` (a broadcast running
    accumulator) comes back as a full-extent slice.
    """
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
    """Insert `tile` back into `dest` at the current reduction tile's offsets.

    The mirror of `_tile_slice`: it yields the loop-carried destination for a
    fused op whose result spans `tiled_dim`. For a result that does not span it
    (a reduction accumulator) this degenerates to a full-extent insert.
    """
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
    """Rebuild `src` over new operands, copying its maps, iterators and body.

    Retiling changes only operand *extents*, never the indexing maps or iterator
    types, so both are carried over verbatim. The body is cloned op by op: a
    linalg body is straight-line scalar arithmetic with no nested regions.
    """
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
    """Emit a ``linalg.elementwise`` of `kind` over `inputs` into `dest`.

    The Python builder fills in the default identity indexing maps but leaves the
    region empty (the C++ region builder is not bound), so the scalar body is emitted
    here via `scalar_op` -- which therefore has to agree with `kind`.
    """
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
    """Elementwise ``extf``/``truncf`` of a whole tensor to `element_type`.

    The caller has established that the conversion exists (`irr.cast_float` accepts
    the pair), so the emitted body cannot fail to build.
    """
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
        linalg.yield_([irr.cast_float(body.arguments[0], element_type)])
    return op.results[0]


def _emit_correction_term(
    e: ir.OpView,
    bindings: list[tuple[ls.Operand, ir.Value]],
    compute_type: ir.Type,
) -> ir.Value | None:
    """Emit `e`'s body at the current insertion point, isolated on its accumulators.

    `bindings` binds each accumulator-reading ``E`` input operand to the scalar to
    substitute for it -- the current-tile value for the "new" factor, the previous
    running value for the "old" one.

    Every op is rebuilt at `compute_type` rather than at the type it had in ``E``,
    so the term is evaluated in the precision `_correction_factor` picked; the bound
    values are already of that type, and anything the body captures from an enclosing
    scope is converted to it.

    Because ``E`` is a separate op its body can be cloned directly; no backward
    slice is needed, ``E``'s yielded value *is* the term. Data block arguments are
    replaced by a stand-in constant chosen per consuming op
    (`operand_eliminating_constant`): ``1.0`` for ``mulf``/``divf``, ``0.0`` for
    ``addf``/``subf``. What makes that sound is not the constant itself but
    separability, which legality has already established: with
    ``E(x, m) = f(x) * g(m)``, substituting any ``c`` for ``x`` leaves
    ``E(c, m_new) / E(c, m_old) = g(m_new) / g(m_old)``. For softmax
    ``E = exp(x - m)`` and ``c = 0`` that is ``exp(-m_new) / exp(-m_old)``, the same
    ratio as ``exp(x - m_new) / exp(x - m_old)`` for any ``x``.

    TODO: Caveat: the cancellation needs ``f(c)`` to be finite and non-zero, which is not
    checked. A body where the substitution zeroes the data factor -- ``mulf`` fed by
    a ``subf`` of two data arguments, say ``(x1 - x2) * exp(-m)`` -- is accepted by
    the separability analysis but yields ``0 / 0`` here.

    Returns the cloned term value, or None if a data argument feeds an op with no
    stand-in constant, or a captured value cannot be converted to `compute_type`.
    """
    body = e.regions[0].blocks[0]
    ops = list(body.operations)
    term = ops[-1].operands[0]

    # Bind the accumulator block arguments; data arguments stay unmapped and are
    # neutralized per consuming op below.
    value_map: dict = {}
    for operand, value in bindings:
        value_map[ls.matching_block_argument(e, operand)] = value

    for op in ops[:-1]:
        ov = opview(op)
        temporary: list = []
        for operand in ov.operands:
            if operand in value_map:
                continue
            if isinstance(operand, ir.BlockArgument) and operand.owner == body:
                neutral = irr.operand_eliminating_constant(ov, compute_type)
                if neutral is None:
                    return None
                value_map[operand] = arith.constant(compute_type, neutral)
                temporary.append(operand)
            else:
                # Captured from an enclosing scope: kept, but at `compute_type`. Not
                # dropped afterwards -- unlike a neutral element it is reusable.
                converted = irr.cast_float(operand, compute_type)
                if converted is None:
                    return None
                value_map[operand] = converted
        if irr.clone_op_with_map(ov, value_map, result_type=compute_type) is None:
            return None
        # Drop the per-op substitutions so the next consumer of the same data
        # argument gets its own neutral element.
        for operand in temporary:
            del value_map[operand]

    return value_map.get(term, term)


def _correction_factor(
    e: ir.OpView,
    r2: ir.OpView,
    accumulators: list[tuple[ls.Operand, ir.Value, ir.Value]],
    r2_accumulator: ir.Value,
) -> ir.Value | None:
    """Build the online rescale factor for the current tile: ``term(new)/term(old)``.

    `accumulators` holds ``(fused E operand, new value, old value)`` per running
    accumulator the fused ``E`` reads. The two ``term`` evaluations and the
    division are ``linalg.generic``s over ``R2``'s *parallel* iteration space, so
    the factor takes ``R2``'s accumulator shape -- for a contraction that space is
    wider than ``E``'s (a GEMM's extra N dim) and the accumulator must broadcast
    over it.

    Each accumulator is read through its own ``E`` indexing map, translated from
    ``E``'s dims into ``R2``'s and with ``R2``'s reduction dim projected out.
    Legality guarantees those maps do not reference the reduction axis, so the
    projection is lossless.

    ``E``'s body and ``R2``'s accumulator need not share an element type (narrow
    probabilities feeding a wide contraction accumulator is the usual attention
    shape), so the term is evaluated in whichever of the two is wider and cast at the
    boundaries: the accumulator inputs are converted on entry to the body, and the
    finished factor is converted back to ``R2``'s accumulator type for the rescale
    multiply. Evaluating in the wider type keeps a ratio of two exponentials off f16's
    exponent range even when the payload's term is f16.
    """
    # Align E's output map with R2's map for E's result position-by-position to
    # get the E-dim -> R2-dim correspondence.
    r2_e_operand = find_r2_elementwise_operand(r2, e)
    e_out_map = ls.indexing_map_for(e, ls.dps_init_operands(e)[0])
    r2_e_map = ls.indexing_map_for(r2, r2_e_operand)
    if len(e_out_map.results) != len(r2_e_map.results):
        return None
    e_dim_to_r2: dict[int, int] = {}
    for e_expr, r2_expr in zip(e_out_map.results, r2_e_map.results):
        if not isinstance(e_expr, ir.AffineDimExpr) or not isinstance(
            r2_expr, ir.AffineDimExpr
        ):
            return None
        e_dim_to_r2[e_expr.position] = r2_expr.position

    r2_red_dim = ls.reduction_dims(r2)[0]
    accumulator_type = ir.RankedTensorType(r2_accumulator.type)
    rank = accumulator_type.rank

    indexing_maps = []
    for operand, _, _ in accumulators:
        in_r2 = irr.remap_dims(
            ls.indexing_map_for(e, operand), e_dim_to_r2, ls.num_loops(r2)
        )
        if in_r2 is None:
            return None
        projected = irr.project_dims(in_r2, {r2_red_dim})
        if projected is None or projected.n_dims != rank:
            return None
        indexing_maps.append(projected)
    # The factor takes R2's accumulator shape, with an identity map over its rank.
    indexing_maps.append(ir.AffineMap.get_identity(rank))

    maps_attr = ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in indexing_maps])
    iterator_types = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )
    element_type = accumulator_type.element_type
    # Legality has already checked the two types have a common widening.
    compute_type = irr.wider_float_type(
        ir.ShapedType(e.results[0].type).element_type, element_type
    )
    if compute_type is None:
        return None
    init = tensor.empty(irr.mixed_sizes(r2_accumulator), compute_type)

    def build_term(pick) -> ir.Value | None:
        inputs = [pick(acc) for acc in accumulators]
        op = linalg.GenericOp(
            result_tensors=[init.type],
            inputs=inputs,
            outputs=[init],
            indexing_maps=maps_attr,
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
                if converted is None:
                    return None
                bindings.append((operand, converted))
            term = _emit_correction_term(e, bindings, compute_type)
            # `E` may yield a bare accumulator or a captured value; either way the
            # yielded type has to be the output's.
            term = None if term is None else irr.cast_float(term, compute_type)
            if term is None:
                return None
            linalg.yield_([term])
        return op.results[0]

    term_new = build_term(lambda acc: acc[1])
    term_old = build_term(lambda acc: acc[2])
    if term_new is None or term_old is None:
        return None

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


def fuse_dependant_reduction_ops(
    rewriter,
    r1_loop: ir.OpView,
    e: ir.OpView,
    r2: ir.OpView,
    e_tiled_dim: int,
    tile_size: int,
) -> ir.OpView:
    """Fuse the ``R1 -> E -> R2`` chain into `r1_loop`, returning the fused loop.

    `e_tiled_dim` is the ``E`` loop dim carrying ``R2``'s reduction axis and
    `tile_size` the loop's step, both as returned by
    ``check_legal_fusion_triple``. Every payload mutation goes through `rewriter`
    so the transform state keeps tracking the ops it replaces.
    """
    r1_loop, e, r2 = opview(r1_loop), opview(e), opview(r2)
    r2_red_dim = ls.reduction_dims(r2)[0]
    num_old_results = len(list(r1_loop.results))

    # Fuse a *clone* of E when another consumer needs the term too, leaving the
    # original in place for them (see `needs_elementwise_clone`). The clone goes
    # immediately before E, which keeps it above every other user of an R1 loop
    # result, so the new loop built at its position still dominates them all.
    fused_source = e
    if needs_elementwise_clone(e, r2):
        r2_e_operand = find_r2_elementwise_operand(r2, e)
        with ir.InsertionPoint(e), e.location:
            clone = opview(e.operation.clone())
        r2_e_operand.set(clone.results[0])
        fused_source = clone

    # The new loop replaces `fused_source`'s position: below E's and R2's inits
    # (so they dominate it) yet at or above every user of an R1 loop result, which
    # legality requires to post-dominate E.
    anchor = fused_source
    e_init_operand = ls.dps_init_operands(fused_source)[0]
    r2_init_operand = ls.dps_init_operands(r2)[0]

    # Hoist the operand definitions the new loop will need above it. Tiling
    # routinely leaves them below: E's and R2's destinations (a `tensor.empty` /
    # `linalg.fill`) and, when an enclosing `scf.forall` tiled a parallel dim, the
    # `tensor.extract_slice` feeding a data input -- e.g. the per-batch slice of
    # `V` in an attention chain. Operands that depend on the loop are skipped;
    # moving them would try to move the loop above itself.
    to_hoist = [
        operand.value
        for consumer in (fused_source, r2)
        for operand in ls.operands_of(consumer)
        if not irr.depends_on_op(operand.value, r1_loop)
    ]
    if not irr.move_value_definitions(to_hoist, anchor):
        raise FusionRejected(
            "could not move the fused ops' operand definitions above the reduction loop"
        )

    e_dest = e_init_operand.value
    r2_dest = r2_init_operand.value
    result_to_inner = map_loop_results_to_inner_reductions(r1_loop)
    accumulator_operands, accumulator_result_indices = collect_r1_as_elementwise_inputs(
        r1_loop, fused_source
    )

    # --- build the replacement loop ------------------------------------------
    with ir.InsertionPoint(anchor), r1_loop.location:
        new_loop = scf.ForOp(
            r1_loop.lowerBound,
            r1_loop.upperBound,
            r1_loop.step,
            list(r1_loop.initArgs) + [e_dest, r2_dest],
        )
    # Carry the original loop's attributes over, `__reduction_loop__` included, so
    # the fused loop is still recognisable as a tiled reduction loop for a
    # subsequent application (which is how a second consumer reduction is fused).
    for name, attr in irr.op_attributes(r1_loop).items():
        new_loop.operation.attributes[name] = attr

    iv = new_loop.induction_variable
    e_arg = new_loop.inner_iter_args[num_old_results]
    r2_arg = new_loop.inner_iter_args[num_old_results + 1]

    value_map: dict = {r1_loop.induction_variable: iv}
    value_map.update(zip(r1_loop.inner_iter_args, new_loop.inner_iter_args))

    old_body_ops = list(r1_loop.body.operations)
    with ir.InsertionPoint(new_loop.body), r1_loop.location:
        # The original body, cloned; its inner reductions now accumulate into the
        # new loop's `iter_arg`s.
        for op in old_body_ops[:-1]:
            irr.clone_op_deep_with_map(op, value_map)

        # --- the fused E, re-sliced to the current tile ---
        # Each accumulator the fused E reads maps to its current-tile value (the
        # cloned inner reduction's result) and its previous running value (that
        # reduction's own DPS init). Using the init rather than the raw `iter_arg`
        # keeps the old value readable *after* the inner reduction has run: the
        # init is an immutable SSA value, so bufferization copies the accumulator
        # instead of letting the reduction write in place.
        accumulator_values = {}
        for operand, result_idx in zip(
            accumulator_operands, accumulator_result_indices
        ):
            inner = result_to_inner[result_idx]
            new_value = value_map[inner.results[0]]
            old_value = value_map.get(
                ls.dps_init_operands(inner)[0].value,
                ls.dps_init_operands(inner)[0].value,
            )
            accumulator_values[operand] = (new_value, old_value)

        e_inputs = []
        for operand in ls.dps_input_operands(fused_source):
            if operand in accumulator_values:
                # A running accumulator: broadcast over the reduction axis, so it
                # is read whole rather than sliced.
                e_inputs.append(accumulator_values[operand][0])
            else:
                e_inputs.append(
                    _tile_slice(
                        operand.value,
                        ls.indexing_map_for(fused_source, operand),
                        e_tiled_dim,
                        iv,
                        tile_size,
                    )
                )
        e_out_map = ls.indexing_map_for(fused_source, e_init_operand)
        e_dest_tile = _tile_slice(e_arg, e_out_map, e_tiled_dim, iv, tile_size)
        fused_e = _clone_generic_with_operands(
            fused_source, e_inputs, [e_dest_tile], [e_dest_tile.type]
        )

        # --- the online correction on R2's running accumulator ---
        r2_out_map = ls.indexing_map_for(r2, r2_init_operand)
        r2_acc_tile = _tile_slice(r2_arg, r2_out_map, r2_red_dim, iv, tile_size)
        accumulators = [
            (operand, new_value, old_value)
            for operand, (new_value, old_value) in accumulator_values.items()
        ]
        if not accumulators:
            raise FusionRejected(
                "fused E does not consume a running accumulator of the loop"
            )
        factor = _correction_factor(fused_source, r2, accumulators, r2_acc_tile)
        if factor is None:
            raise FusionRejected("could not build the correction factor from E")
        scaled = _emit_elementwise(
            linalg.ElementwiseKind.mul,
            lambda a, b: arith.mulf(a, b),
            [r2_acc_tile, factor],
            r2_acc_tile,
        )

        # --- the fused R2, accumulating this tile into the rescaled sum ---
        e_result = fused_source.results[0]
        r2_inputs = []
        for operand in ls.dps_input_operands(r2):
            if operand.value == e_result:
                r2_inputs.append(fused_e.results[0])
            else:
                r2_inputs.append(
                    _tile_slice(
                        operand.value,
                        ls.indexing_map_for(r2, operand),
                        r2_red_dim,
                        iv,
                        tile_size,
                    )
                )
        fused_r2 = _clone_generic_with_operands(
            r2, r2_inputs, [scaled], [r2.results[0].type]
        )

        # --- the new yield ---
        yielded = [value_map[o] for o in old_body_ops[-1].operands]
        yielded.append(
            _insert_tile_slice(
                fused_e.results[0], e_arg, e_out_map, e_tiled_dim, iv, tile_size
            )
        )
        yielded.append(
            _insert_tile_slice(
                fused_r2.results[0], r2_arg, r2_out_map, r2_red_dim, iv, tile_size
            )
        )
        scf.YieldOp(yielded)

    # --- retire the originals -------------------------------------------------
    # R2 first, then E: erasing R2 drops the only use of the fused E's full-extent
    # result, leaving the loop result that replaces it dead (as it must be -- it
    # holds a stale term in every tile but the last).
    rewriter.replace_op(r2, [new_loop.results[num_old_results + 1]])
    rewriter.replace_op(fused_source, [new_loop.results[num_old_results]])
    rewriter.replace_op(r1_loop, list(new_loop.results)[:num_old_results])
    return new_loop

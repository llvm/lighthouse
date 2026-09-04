"""Transform extension to generate fused attention computation at tensor level."""

from mlir import ir
from mlir.dialects import ext, transform, arith, scf, math, linalg, tensor
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _scalar_constant(value, element_type):
    """Emit a scalar arith.constant of the given float type."""
    return arith.constant(element_type, ir.FloatAttr.get(element_type, value))


def _empty(shape, element_type):
    """Emit a tensor.empty of the given static shape."""
    return tensor.empty(list(shape), element_type)


def _filled(shape, element_type, value):
    """Emit a tensor.empty initialized with `value` through a linalg.fill."""
    return linalg.fill(
        _scalar_constant(value, element_type), outs=[_empty(shape, element_type)]
    )


def _iterators(num_parallel, num_reduction):
    """Build a linalg iterator_types array attribute."""
    parallel = ir.Attribute.parse("#linalg.iterator_type<parallel>")
    reduction = ir.Attribute.parse("#linalg.iterator_type<reduction>")
    return ir.ArrayAttr.get([parallel] * num_parallel + [reduction] * num_reduction)


def _generic(ins, outs, maps, iterators, body):
    """Emit a single-result linalg.generic.

    `maps` holds the indexing maps of `ins` followed by those of `outs`,
    `iterators` the iterator_types attribute and `body` a callable receiving the
    block arguments (inputs then outputs) and returning the value to yield.
    """
    generic = linalg.GenericOp(
        result_tensors=[out.type for out in outs],
        inputs=ins,
        outputs=outs,
        indexing_maps=ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in maps]),
        iterator_types=iterators,
    )
    block = generic.regions[0].blocks.append(
        *[
            ir.ShapedType(operand.type).element_type
            for operand in list(ins) + list(outs)
        ]
    )
    with ir.InsertionPoint(block):
        linalg.yield_([body(*block.arguments)])
    return generic.results[0]


def _tile_and_row_maps(nb):
    """Maps over the [*batch, m, trailing] iteration space (nb + 2 dims).

    Returns the identity map onto the full tile and the map dropping the
    trailing dim, i.e. the one used by the [*batch, m] row vectors (either as
    reduction results or as operands broadcast over the trailing dim).
    """
    dims = [ir.AffineExpr.get_dim(i) for i in range(nb + 2)]
    return ir.AffineMap.get(nb + 2, 0, dims), ir.AffineMap.get(nb + 2, 0, dims[:-1])


def _contract(lhs, rhs, acc, batch_shape):
    """Emit a batched contraction `acc += lhs @ rhs` as a linalg.contract.

    lhs is [*batch, m, k], rhs is [*batch, k, n] and acc is [*batch, m, n],
    reducing over k. linalg.contract casts operands narrower than `acc` up to its
    element type, which the vectorizer then folds back into the resulting
    vector.contract (see `fold_type_extensions_into_contract`), so the DPAS keeps
    its narrow operands and wide accumulator.
    """
    nb = len(batch_shape)
    dims = [ir.AffineExpr.get_dim(i) for i in range(nb + 3)]
    batch = dims[:nb]
    m, n, k = dims[nb], dims[nb + 1], dims[nb + 2]

    return linalg.contract(
        lhs,
        rhs,
        outs=[acc],
        indexing_maps=[
            ir.AffineMap.get(nb + 3, 0, batch + [m, k]),
            ir.AffineMap.get(nb + 3, 0, batch + [k, n]),
            ir.AffineMap.get(nb + 3, 0, batch + [m, n]),
        ],
    )


def _row_reduce(source, acc, combiner, batch_shape):
    """Reduce a [*batch, m, trailing] tile to [*batch, m] over the trailing dim."""
    nb = len(batch_shape)
    tile_map, row_map = _tile_and_row_maps(nb)
    return _generic(
        [source],
        [acc],
        [tile_map, row_map],
        _iterators(nb + 1, 1),
        combiner,
    )


def _elemwise(ins, out, batch_shape, body, broadcast_row_operands=()):
    """Emit an elementwise linalg.generic writing into `out`.

    `out` is either a [*batch, m] row vector or a [*batch, m, trailing] tile.
    Operands listed in `broadcast_row_operands` (by index into `ins`) are row
    vectors broadcast over the trailing dim of `out`.
    """
    nb = len(batch_shape)
    is_tile = ir.ShapedType(out.type).rank == nb + 2
    if is_tile:
        out_map, row_map = _tile_and_row_maps(nb)
    else:
        out_map = row_map = ir.AffineMap.get_identity(nb + 1)
    in_maps = [
        row_map if i in broadcast_row_operands else out_map for i in range(len(ins))
    ]
    return _generic(
        ins,
        [out],
        in_maps + [out_map],
        _iterators(nb + 2 if is_tile else nb + 1, 0),
        body,
    )


def _drop_leading_dims(source, num_dims):
    """Rank-reduce `source` by slicing away its `num_dims` leading unit dims."""
    source_type = ir.RankedTensorType(source.type)
    sizes = list(source_type.shape)
    return tensor.ExtractSliceOp(
        ir.RankedTensorType.get(sizes[num_dims:], source_type.element_type),
        source,
        [],
        [],
        [],
        [0] * source_type.rank,
        sizes,
        [1] * source_type.rank,
    ).result


def _restore_leading_dims(source, destination):
    """Insert a rank-reduced `source` back under the leading unit dims of `destination`."""
    destination_type = ir.RankedTensorType(destination.type)
    return tensor.InsertSliceOp(
        source,
        destination,
        [],
        [],
        [],
        [0] * destination_type.rank,
        list(destination_type.shape),
        [1] * destination_type.rank,
    ).result


def _extract_kv_tile(source, loop_idx, tile_size):
    """Slice a [*batch, n_ctx, d_head] K/V tensor down to `tile_size` rows.

    Extracts [*batch, tile_size, d_head] starting at `loop_idx` along the n_ctx
    dim, i.e. at offset [0, ..., loop_idx, 0].
    """
    source_type = ir.RankedTensorType(source.type)
    rank = source_type.rank
    sizes = list(source_type.shape)
    sizes[-2] = tile_size

    static_offsets = [0] * rank
    static_offsets[-2] = ir.ShapedType.get_dynamic_size()

    return tensor.ExtractSliceOp(
        ir.RankedTensorType.get(sizes, source_type.element_type),
        source,
        [loop_idx],
        [],
        [],
        static_offsets,
        sizes,
        [1] * rank,
    ).result


class ReplaceWithFusedAttentionOp(
    TransformExtensionDialect.Operation, name="generate_fused_attention"
):
    """Replace a tensor-level attention output with a fused (flash) attention loop.

    Takes the Q, K, V tensors and the scale constant of a tiled but not yet
    vectorized attention region and replaces the P@V linalg contraction with an
    `scf.for` over the K/V sequence length that computes the same result with
    online softmax:

        m, l, acc = -inf, 0, 0
        for j in range(0, n_ctx, tile_size):
            s     = (Q @ K[j:j+tile]^T) * scale
            m_new = max(m, rowmax(s))
            p     = exp(s - m_new)
            alpha = exp(m - m_new)
            l     = l * alpha + rowsum(p)
            acc   = acc * alpha + p @ V[j:j+tile]
            m     = m_new
        out = acc / l

    Doing this at tensor level keeps the tiling and fusion decisions at the
    level where the rest of the schedule makes them; the regular vectorization
    stage then lowers the emitted loop. The dead softmax and Q@K^T producers
    left behind are removed by DCE.

    Args:
        q: Handle to the op producing the Q tile [*batch, wg_rows, d_head]
        k: Handle to the op producing the K tensor [*batch, n_ctx, d_head]
        v: Handle to the op producing the V tensor [*batch, n_ctx, d_head]
        scale: Handle to the scale constant op (scalar arith.constant)
        output: Handle to the P@V linalg contraction to replace
        tile_size: Tile size for the reduction dimension (K/V sequence length)
    """

    q: ext.Operand[transform.AnyOpType]
    k: ext.Operand[transform.AnyOpType]
    v: ext.Operand[transform.AnyOpType]
    scale: ext.Operand[transform.AnyOpType]
    output: ext.Operand[transform.AnyOpType]
    tile_size: ir.IntegerAttr
    new_output: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, ctx=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=ctx)

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "ReplaceWithFusedAttentionOp",
            rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            payloads = []
            for handle in (op.q, op.k, op.v, op.scale, op.output):
                handle_ops = state.get_payload_ops(handle)
                if len(handle_ops) != 1:
                    return DiagnosedSilenceableFailure.emit_silenceable_error(
                        "Expected exactly one operation for each operand"
                    )
                payloads.append(handle_ops[0])
            q_op, k_op, v_op, scale_op, output_op = payloads

            if not isinstance(scale_op.opview, arith.ConstantOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected scale to be arith.constant, got {scale_op.name}"
                )
            if not output_op.name.startswith("linalg."):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected output to be a linalg op, got {output_op.name}"
                )

            q, k, v = (payload.results[0] for payload in (q_op, k_op, v_op))
            for name, value in (("q", q), ("k", k), ("v", v)):
                if not isinstance(value.type, ir.RankedTensorType):
                    return DiagnosedSilenceableFailure.emit_silenceable_error(
                        f"Expected {name} to produce a ranked tensor, got {value.type}"
                    )

            # The last two dims of Q are [wg_rows(M), d_head]; any leading dims
            # are batch dims carried through unchanged. Nothing is assumed about
            # the rank, so batched and non-batched payloads both work.
            q_type = ir.RankedTensorType(q.type)
            batch_shape = list(q_type.shape[:-2])
            wg_rows, d_head = q_type.shape[-2], q_type.shape[-1]
            n_ctx = ir.RankedTensorType(k.type).shape[-2]
            tile_size = ir.IntegerAttr(op.tile_size).value

            # Element types are read from the matched ops since Q, K, V and the
            # softmax weights (P) may each use a different (possibly mixed)
            # precision. Both matmul accumulators and the online softmax run in
            # f32 for numerical accuracy; only the matmul operands keep their
            # narrower element types.
            k_element_type = ir.RankedTensorType(k.type).element_type
            # P is the lhs of the `P @ V` contraction, so it carries V's element
            # type: a `linalg.batch_matmul`'s two operands must agree. Derived from
            # `v` rather than from `output_op`'s lhs so that `output` may be either
            # the contraction itself or, when the payload defers the normalizing
            # divide past it, that divide.
            p_element_type = ir.RankedTensorType(v.type).element_type
            out_element_type = ir.RankedTensorType(
                output_op.results[0].type
            ).element_type
            compute_type = ir.F32Type.get()

            scale_value = ir.FloatAttr(scale_op.attributes["value"]).value

            # WG tiling leaves the batch dims at extent one. Slice them away, as
            # the XeGPU layout propagation cannot distribute the rank-3
            # broadcasts and reductions of the online softmax.
            squeeze = len(batch_shape) if all(d == 1 for d in batch_shape) else 0
            if squeeze:
                batch_shape = []

            row_shape = (*batch_shape, wg_rows)
            acc_shape = (*batch_shape, wg_rows, d_head)
            qkt_shape = (*batch_shape, wg_rows, tile_size)
            nb = len(batch_shape)

            with ir.InsertionPoint(output_op):
                if squeeze:
                    q, k, v = (_drop_leading_dims(t, squeeze) for t in (q, k, v))

                scale_tile = _filled(qkt_shape, compute_type, scale_value)

                index_type = ir.IndexType.get()
                loop = scf.ForOp(
                    arith.constant(index_type, 0),
                    arith.constant(index_type, n_ctx),
                    arith.constant(index_type, tile_size),
                    [
                        _filled(row_shape, compute_type, float("-inf")),
                        _filled(row_shape, compute_type, 0.0),
                        _filled(acc_shape, compute_type, 0.0),
                    ],
                )

                with ir.InsertionPoint(loop.body):
                    loop_idx = loop.induction_variable
                    m_i, l_i, acc = loop.inner_iter_args

                    # S = (Q @ K[j:j+tile]^T) * scale, accumulated in f32. K is
                    # transposed explicitly so that the vectorized contraction
                    # takes the same operand layouts as the untiled kernel.
                    k_tile = _extract_kv_tile(k, loop_idx, tile_size)
                    k_transposed = linalg.transpose(
                        k_tile,
                        outs=[
                            _empty((*batch_shape, d_head, tile_size), k_element_type)
                        ],
                        permutation=[*range(nb), nb + 1, nb],
                    ).results[0]
                    qkt = _contract(
                        q,
                        k_transposed,
                        _filled(qkt_shape, compute_type, 0.0),
                        batch_shape,
                    )
                    qkt_scaled = _elemwise(
                        [qkt, scale_tile],
                        _empty(qkt_shape, compute_type),
                        batch_shape,
                        lambda a, b, out: arith.mulf(a, b),
                    )

                    # m_new = max(m, rowmax(S)), accumulated straight into m so
                    # that the carried value stays a register once vectorized.
                    m_new = _row_reduce(
                        qkt_scaled,
                        m_i,
                        lambda value, out: arith.maximumf(value, out),
                        batch_shape,
                    )

                    # P = exp(S - m_new). fastmath<fast> lets the exp lower to
                    # the native hardware exp; without it the accurate expansion
                    # doubles the exp count and scalarizes part of it.
                    p = _elemwise(
                        [qkt_scaled, m_new],
                        _empty(qkt_shape, compute_type),
                        batch_shape,
                        lambda a, b, out: math.exp(arith.subf(a, b), fastmath="fast"),
                        broadcast_row_operands={1},
                    )

                    # alpha = exp(m - m_new) rescales the running row sum and
                    # the P@V accumulator to the new row maximum.
                    alpha = _elemwise(
                        [m_i, m_new],
                        _empty(row_shape, compute_type),
                        batch_shape,
                        lambda a, b, out: math.exp(arith.subf(a, b), fastmath="fast"),
                    )

                    # l = l * alpha + rowsum(P)
                    l_scaled = _elemwise(
                        [l_i, alpha],
                        l_i,
                        batch_shape,
                        lambda a, b, out: arith.mulf(a, b),
                    )
                    l_new = _row_reduce(
                        p,
                        l_scaled,
                        lambda value, out: arith.addf(value, out),
                        batch_shape,
                    )

                    # acc = acc * alpha + P @ V[j:j+tile], with P narrowed to
                    # the dtype the replaced contraction expects.
                    acc_scaled = _elemwise(
                        [acc, alpha],
                        acc,
                        batch_shape,
                        lambda a, b, out: arith.mulf(a, b),
                        broadcast_row_operands={1},
                    )
                    p_operand = p
                    if p_element_type != compute_type:
                        p_operand = _elemwise(
                            [p],
                            _empty(qkt_shape, p_element_type),
                            batch_shape,
                            lambda value, out: arith.truncf(p_element_type, value),
                        )
                    v_tile = _extract_kv_tile(v, loop_idx, tile_size)
                    acc_new = _contract(p_operand, v_tile, acc_scaled, batch_shape)

                    scf.yield_([m_new, l_new, acc_new])

            # out = acc / l, narrowed back to the element type of the replaced
            # op. Its destination is reused so that bufferization writes the
            # result in place; the destination's now dead zero fill is dropped
            # by DCE.
            destination = output_op.operands[-1]
            if isinstance(destination, ir.OpResult) and isinstance(
                destination.owner, linalg.FillOp
            ):
                destination = destination.owner.operands[1]

            with ir.InsertionPoint.after(loop):
                _, l_out, acc_out = loop.results

                def normalize(value, row_sum, out):
                    normalized = arith.divf(value, row_sum)
                    if out_element_type != compute_type:
                        normalized = arith.truncf(out_element_type, normalized)
                    return normalized

                output_final = _elemwise(
                    [acc_out, l_out],
                    _drop_leading_dims(destination, squeeze)
                    if squeeze
                    else destination,
                    batch_shape,
                    normalize,
                    broadcast_row_operands={1},
                )
                if squeeze:
                    output_final = _restore_leading_dims(output_final, destination)

            output_op.results[0].replace_all_uses_with(output_final)
            rewriter.erase_op(output_op)

            results.set_ops(op.new_output, [output_final.owner])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ReplaceWithFusedAttentionOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                # Read Q, K, V and scale
                transform.only_reads_handle(op.op_operands[:4])
                # Consume and replace output
                + transform.consumes_handle(op.op_operands[4:5])
                # Produce new output handle
                + transform.produces_handle(op.results)
                # Modify the payload
                + transform.modifies_payload()
            )


def replace_with_fused_attention(
    q: ir.Value,
    k: ir.Value,
    v: ir.Value,
    scale: ir.Value,
    output: ir.Value,
    tile_size: int | ir.IntegerAttr,
) -> ir.Value:
    """Replace a tensor-level attention output with a fused attention loop.

    Args:
        q: Handle to the op producing the Q tile [*batch, wg_rows, d_head]
        k: Handle to the op producing the K tensor [*batch, n_ctx, d_head]
        v: Handle to the op producing the V tensor [*batch, n_ctx, d_head]
        scale: Handle to the scale constant op (scalar arith.constant)
        output: Handle to the P@V linalg contraction to replace
        tile_size: Tile size for the reduction dimension (K/V sequence length)

    Returns:
        Handle to the new output operation
    """
    if not isinstance(tile_size, ir.IntegerAttr):
        tile_size = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)

    return ReplaceWithFusedAttentionOp(
        q, k, v, scale, output, tile_size=tile_size
    ).new_output

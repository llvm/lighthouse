"""Transform extension to generate fused attention computation."""

import numpy as np
from mlir import ir
from mlir.dialects import ext, transform, arith, scf, math, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure
from lighthouse.utils.numpy import mlir_to_numpy_dtype

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def emit_vector_constant(shape, fill_value, element_type):
    """Emit an arith.constant of vector type, filled with fill_value."""
    vector_type = ir.VectorType.get(list(shape), element_type)
    np_dtype = mlir_to_numpy_dtype(element_type)
    values = np.full(shape, fill_value, dtype=np_dtype)
    attr = ir.DenseElementsAttr.get(values, type=vector_type)
    return arith.constant(vector_type, attr)


def _iterator_types(num_parallel, num_reduction):
    """Build a vector.contract iterator_types array attribute."""
    parallel = ir.Attribute.parse("#vector.iterator_type<parallel>")
    reduction = ir.Attribute.parse("#vector.iterator_type<reduction>")
    return ir.ArrayAttr.get([parallel] * num_parallel + [reduction] * num_reduction)


def _broadcast_last_dim(value, batch_shape, wg_rows, last_dim, element_type):
    """Broadcast a [*batch, wg_rows] vector to [*batch, wg_rows, last_dim].

    vector.broadcast can only prepend leading dims, so the new dimension is
    broadcast to the front and then transposed to the trailing position.
    """
    nb = len(batch_shape)
    bcasted_type = ir.VectorType.get([last_dim, *batch_shape, wg_rows], element_type)
    bcasted = vector.broadcast(bcasted_type, value)
    # Move the leading (last_dim) axis to the back, keeping batch and wg_rows order.
    perm = list(range(1, nb + 2)) + [0]
    out_type = ir.VectorType.get([*batch_shape, wg_rows, last_dim], element_type)
    return vector.transpose(out_type, bcasted, perm)


def compute_qkt(
    q_value,
    k_load_op,
    loop_idx,
    batch_shape,
    wg_rows,
    d_head,
    tile_size,
    k_element_type,
    compute_type,
):
    """Load the K tile, transpose it, and contract with Q to produce Q@K^T.

    The K tile is [*batch, tile_size, d_head], transposed to
    [*batch, d_head, tile_size] and contracted with q_value
    [*batch, wg_rows, d_head] to produce [*batch, wg_rows, tile_size], reducing
    over d_head. K is `k_element_type` (Q keeps whatever type q_value already
    has), the contraction accumulates in `compute_type`.
    """
    nb = len(batch_shape)
    k_memref = k_load_op.operands[0]
    k_load_indices = list(k_load_op.operands[1:-1])
    padding = k_load_op.operands[-1]
    in_bounds = k_load_op.attributes.get("in_bounds", None)
    k_perm_map = k_load_op.attributes.get("permutation_map", None)

    dims = [ir.AffineExpr.get_dim(i) for i in range(nb + 3)]
    batch = dims[:nb]
    m, tile, k = dims[nb], dims[nb + 1], dims[nb + 2]

    q_map = ir.AffineMap.get(nb + 3, 0, batch + [m, k])
    k_map = ir.AffineMap.get(nb + 3, 0, batch + [k, tile])
    out_map = ir.AffineMap.get(nb + 3, 0, batch + [m, tile])

    indexing_maps = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(q_map),
            ir.AffineMapAttr.get(k_map),
            ir.AffineMapAttr.get(out_map),
        ]
    )
    iterator_types = _iterator_types(nb + 2, 1)

    qkt_type = ir.VectorType.get([*batch_shape, wg_rows, tile_size], compute_type)
    qkt_acc = emit_vector_constant(
        (*batch_shape, wg_rows, tile_size), 0.0, compute_type
    )

    k_tile_indices = k_load_indices.copy()
    k_tile_indices[-2] = loop_idx

    k_tile_type = ir.VectorType.get([*batch_shape, tile_size, d_head], k_element_type)
    k_tile = vector.TransferReadOp(
        k_tile_type,
        k_memref,
        k_tile_indices,
        k_perm_map,
        padding,
        in_bounds=in_bounds,
    ).result

    k_transpose_type = ir.VectorType.get(
        [*batch_shape, d_head, tile_size], k_element_type
    )
    k_transpose_perm = list(range(nb)) + [nb + 1, nb]
    k_transpose = vector.transpose(k_transpose_type, k_tile, k_transpose_perm)

    return vector.contract(
        qkt_type,
        q_value,
        k_transpose,
        qkt_acc,
        indexing_maps=indexing_maps,
        iterator_types=iterator_types,
    )


def compute_online_softmax_and_sum(
    qkt_scaled,
    m_ij,
    l_i_init,
    batch_shape,
    wg_rows,
    tile_size,
    element_type,
):
    """Apply online softmax to the scaled Q@K^T and reduce to a row-wise sum.

    Computes exp(qkt_scaled - m_ij), with m_ij broadcast over the inner dim.
    Returns (qkt_exp, l_ij) where qkt_exp is the [*batch, wg_rows, tile_size] exp
    tile and l_ij is its row-wise sum [*batch, wg_rows] (added into l_i_init).
    """
    nb = len(batch_shape)
    m_ij_bcasted = _broadcast_last_dim(
        m_ij, batch_shape, wg_rows, tile_size, element_type
    )

    qkt_centered = arith.subf(qkt_scaled, m_ij_bcasted)
    # fastmath<fast> lets the exp lower to the native hardware exp; without it
    # the accurate expansion doubles the exp count and scalarizes part of it.
    qkt_exp = math.exp(qkt_centered, fastmath="fast")

    l_ij = vector.multi_reduction(
        kind="add",
        source=qkt_exp,
        acc=l_i_init,
        reduction_dims=[nb + 1],
    )

    return qkt_exp, l_ij


def rescale_pv_out_accumulator(acc, alpha, batch_shape, wg_rows, d_head, compute_type):
    """Rescale the running P@V accumulator by broadcasting alpha across d_head.

    Broadcasts alpha [*batch, wg_rows] to [*batch, wg_rows, d_head] and
    multiplies acc by it elementwise. Returns the rescaled accumulator.
    """
    alpha_bcasted = _broadcast_last_dim(
        alpha, batch_shape, wg_rows, d_head, compute_type
    )
    return arith.mulf(acc, alpha_bcasted)


def compute_pv(
    qkt_exp,
    v_load_op,
    pv_init,
    loop_idx,
    batch_shape,
    acc_vector_type,
    d_head,
    tile_size,
    v_element_type,
):
    """Load the V tile and contract it with the softmax tile, accumulating into pv_init.

    Loads V [*batch, tile_size, d_head] (`v_element_type`) and contracts it with
    the exp tile [*batch, wg_rows, tile_size] (already narrowed to the matching P
    dtype by the caller) into the running [*batch, wg_rows, d_head] accumulator,
    whose type is given by `acc_vector_type`. Returns the accumulated result.
    """
    nb = len(batch_shape)
    v_memref = v_load_op.operands[0]
    v_load_indices = list(v_load_op.operands[1:-1])
    v_padding = v_load_op.operands[-1]
    v_in_bounds = v_load_op.attributes.get("in_bounds", None)
    v_perm_map = v_load_op.attributes.get("permutation_map", None)

    dims = [ir.AffineExpr.get_dim(i) for i in range(nb + 3)]
    batch = dims[:nb]
    m, k, tile = dims[nb], dims[nb + 1], dims[nb + 2]

    qkt_exp_map = ir.AffineMap.get(nb + 3, 0, batch + [m, tile])
    v_map = ir.AffineMap.get(nb + 3, 0, batch + [tile, k])
    pv_out_map = ir.AffineMap.get(nb + 3, 0, batch + [m, k])

    indexing_maps_pv = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(qkt_exp_map),
            ir.AffineMapAttr.get(v_map),
            ir.AffineMapAttr.get(pv_out_map),
        ]
    )
    iterator_types_pv = _iterator_types(nb + 2, 1)

    v_tile_indices = v_load_indices.copy()
    v_tile_indices[-2] = loop_idx

    v_tile_type = ir.VectorType.get([*batch_shape, tile_size, d_head], v_element_type)
    v_tile = vector.TransferReadOp(
        v_tile_type,
        v_memref,
        v_tile_indices,
        v_perm_map,
        v_padding,
        in_bounds=v_in_bounds,
    ).result

    return vector.contract(
        acc_vector_type,
        qkt_exp,
        v_tile,
        pv_init,
        indexing_maps=indexing_maps_pv,
        iterator_types=iterator_types_pv,
    )


def normalize_output_by_sum(
    pv_out, l_i_out, batch_shape, wg_rows, d_head, compute_type
):
    """Divide pv_out [*batch, wg_rows, d_head] by l_i_out [*batch, wg_rows]."""
    l_i_out_bcasted = _broadcast_last_dim(
        l_i_out, batch_shape, wg_rows, d_head, compute_type
    )
    return arith.divf(pv_out, l_i_out_bcasted)


class ReplaceWithFusedAttentionOp(
    TransformExtensionDialect.Operation, name="generate_fused_attention"
):
    """Replace a given (standard) attention output with an equivalent output that is
    computed in a fused fashion (fused attention optimization).

    Takes Q, K, V loads and scale constant from bufferized IR, and generates an inner
    tiled loop that computes fused attention with online softmax using running max and sum.

    This implements the flash attention algorithm where:
    1. The computation is tiled along the reduction dimension (K/V sequence length)
    2. Online max and sum are maintained across tiles
    3. Output is incrementally updated with rescaled contributions

    Args:
        q_load: Handle to Q load operation (vector.transfer_read)
        k_load: Handle to K load operation (vector.transfer_read)
        v_load: Handle to V load operation (vector.transfer_read)
        scale: Handle to scale constant operation (arith.constant)
        output: Handle to the output operation to replace (vector.contract)
        tile_size: Tile size for the reduction dimension tiling (K/V sequence length)
    """

    q_load: ext.Operand[transform.AnyOpType]
    k_load: ext.Operand[transform.AnyOpType]
    v_load: ext.Operand[transform.AnyOpType]
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
            # Get payload operations
            q_load_ops = state.get_payload_ops(op.q_load)
            k_load_ops = state.get_payload_ops(op.k_load)
            v_load_ops = state.get_payload_ops(op.v_load)
            scale_ops = state.get_payload_ops(op.scale)
            output_ops = state.get_payload_ops(op.output)

            if (
                len(q_load_ops) != 1
                or len(k_load_ops) != 1
                or len(v_load_ops) != 1
                or len(scale_ops) != 1
                or len(output_ops) != 1
            ):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    "Expected exactly one operation for each operand"
                )

            q_load_op = q_load_ops[0]
            k_load_op = k_load_ops[0]
            v_load_op = v_load_ops[0]
            scale_op = scale_ops[0]
            output_op = output_ops[0]

            # Verify operation types
            if not isinstance(q_load_op.opview, vector.TransferReadOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected q_load to be vector.transfer_read, got {q_load_op.operation.name}"
                )
            if not isinstance(k_load_op.opview, vector.TransferReadOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected k_load to be vector.transfer_read, got {k_load_op.operation.name}"
                )
            if not isinstance(v_load_op.opview, vector.TransferReadOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected v_load to be vector.transfer_read, got {v_load_op.operation.name}"
                )
            if not isinstance(scale_op.opview, arith.ConstantOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected scale to be arith.constant, got {scale_op.operation.name}"
                )
            if not isinstance(output_op.opview, vector.ContractionOp):
                return DiagnosedSilenceableFailure.emit_silenceable_error(
                    f"Expected output to be vector.contract, got {output_op.operation.name}"
                )

            # Extract the scale scalar value from scale_op (arith.constant); the
            # splat value avoids materializing a numpy array (which mishandles bf16)
            scale_attr = scale_op.attributes["value"]
            scale_dense_attr = ir.DenseElementsAttr(scale_attr)
            scale_value = ir.FloatAttr(scale_dense_attr.get_splat_value()).value

            # The last two dims of Q are [wg_rows(M), d_head]; any leading dims
            # are batch dims carried through unchanged. Nothing is assumed about
            # the rank, so batched and non-batched payloads both work.
            q_load_result = q_load_op.results[0]
            q_vector_type = ir.VectorType(q_load_result.type)
            batch_shape = list(q_vector_type.shape[:-2])
            wg_rows = q_vector_type.shape[-2]
            d_head = q_vector_type.shape[-1]

            # Get tile size
            tile_size_value = ir.IntegerAttr(op.tile_size).value

            # Element types are read from the actual ops since Q, K, V, and the
            # softmax weights (P) may each use a different (possibly mixed)
            # precision. Q keeps whatever type q_value already has.
            k_element_type = ir.VectorType(k_load_op.results[0].type).element_type
            v_element_type = ir.VectorType(v_load_op.results[0].type).element_type
            # P is the lhs operand of the P@V contract being replaced, so its
            # existing type tells us the precision the rest of the graph expects.
            p_element_type = ir.VectorType(output_op.operands[0].type).element_type
            # Both matmul accumulators and the online softmax (scale, running
            # max and sum, exp) run in f32 for numerical accuracy; only the
            # matmul operands (Q, K, V, P) keep their narrower element types.
            compute_type = ir.F32Type.get()
            # Change this to a narrower type to run the softmax in lower precision;
            # the `reduction_type != compute_type` guards handle the trunc/ext.
            reduction_type = compute_type

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # Define m_i_init: [*batch, wg_rows] with neg_inf values
                m_i_init = emit_vector_constant(
                    (*batch_shape, wg_rows), float("-inf"), reduction_type
                )

                # Define l_i_init: [*batch, wg_rows] with zero values
                l_i_init = emit_vector_constant(
                    (*batch_shape, wg_rows), 0.0, reduction_type
                )

                # Define acc_init: [*batch, wg_rows, d_head] with zero values
                acc_vector_type = ir.VectorType.get(
                    [*batch_shape, wg_rows, d_head], compute_type
                )
                acc_init = emit_vector_constant(
                    (*batch_shape, wg_rows, d_head), 0.0, compute_type
                )

                # Get n_ctx (K/V sequence length) from the second-to-last k dim
                k_load_result = k_load_op.results[0]
                k_vector_type = ir.VectorType(k_load_result.type)
                n_ctx = k_vector_type.shape[-2]
                # Define scale tile: [*batch, wg_rows, tile_size] with the scale value
                scale_tile = emit_vector_constant(
                    (*batch_shape, wg_rows, tile_size_value),
                    scale_value,
                    reduction_type,
                )

                # Create loop bounds
                index_type = ir.IndexType.get()
                c0 = arith.constant(index_type, 0)
                c_n_ctx = arith.constant(index_type, n_ctx)
                c_tile_size = arith.constant(index_type, tile_size_value)

                # Create scf.for loop that iterates from 0 to n_ctx in steps of tile_size
                loop = scf.ForOp(
                    c0, c_n_ctx, c_tile_size, [m_i_init, l_i_init, acc_init]
                )

                with ir.InsertionPoint(loop.body):
                    # Get the loop induction variable and iter_args
                    loop_idx = loop.induction_variable
                    m_i = loop.inner_iter_args[0]
                    l_i = loop.inner_iter_args[1]
                    acc = loop.inner_iter_args[2]

                    q_value = q_load_op.results[0]

                    # Load the K tile, transpose it, and contract with Q to get Q@K^T
                    qkt = compute_qkt(
                        q_value,
                        k_load_op,
                        loop_idx,
                        batch_shape,
                        wg_rows,
                        d_head,
                        tile_size_value,
                        k_element_type,
                        compute_type,
                    )
                    # Truncate Q@K^T (f32 accumulator) to the softmax type before scaling.
                    if reduction_type != compute_type:
                        qkt_narrow_type = ir.VectorType.get(
                            [*batch_shape, wg_rows, tile_size_value], reduction_type
                        )
                        qkt = arith.truncf(qkt_narrow_type, qkt)
                    qkt_scaled = arith.mulf(qkt, scale_tile)

                    # Reduce the scaled Q@K^T to a row-wise max: [*batch, wg_rows]
                    qkt_row_max = vector.multi_reduction(
                        kind="maximumf",
                        source=qkt_scaled,
                        acc=m_i_init,
                        reduction_dims=[len(batch_shape) + 1],
                    )

                    # Compute m_ij = max(m_i, qkt_row_max)
                    # Both have shape [*batch, wg_rows]
                    m_ij = arith.maximumf(m_i, qkt_row_max)

                    # Apply online softmax and reduce to row-wise sum
                    qkt_exp, l_ij = compute_online_softmax_and_sum(
                        qkt_scaled,
                        m_ij,
                        l_i_init,
                        batch_shape,
                        wg_rows,
                        tile_size_value,
                        reduction_type,
                    )

                    # Compute alpha = exp(m_i - m_ij)
                    m_diff = arith.subf(m_i, m_ij)
                    alpha = math.exp(m_diff, fastmath="fast")

                    # Update l_i: l_i_updated = l_i * alpha + l_ij
                    l_i_scaled = arith.mulf(l_i, alpha)
                    l_i_updated = arith.addf(l_i_scaled, l_ij)

                    # Rescale running P@V accumulator by alpha; the accumulator is
                    # kept in f32, so widen the softmax-type alpha to match.
                    alpha_wide = alpha
                    if reduction_type != compute_type:
                        alpha_wide = arith.extf(
                            ir.VectorType.get([*batch_shape, wg_rows], compute_type),
                            alpha,
                        )
                    acc_updated = rescale_pv_out_accumulator(
                        acc, alpha_wide, batch_shape, wg_rows, d_head, compute_type
                    )

                    # Narrow the softmax tile to the dtype the P@V contract expects
                    if reduction_type != p_element_type:
                        qkt_exp_type = ir.VectorType.get(
                            [*batch_shape, wg_rows, tile_size_value], p_element_type
                        )
                        qkt_exp_narrow = arith.truncf(qkt_exp_type, qkt_exp)
                    else:
                        qkt_exp_narrow = qkt_exp

                    # Load the V tile and contract with the softmax tile into pv_out
                    pv_out = compute_pv(
                        qkt_exp_narrow,
                        v_load_op,
                        acc_updated,
                        loop_idx,
                        batch_shape,
                        acc_vector_type,
                        d_head,
                        tile_size_value,
                        v_element_type,
                    )

                    # Yield the updated iter args
                    scf.yield_([m_ij, l_i_updated, pv_out])

            # Extract the final accumulator result (3rd output) from the loop
            pv_out = loop.results[2]
            l_i_out = loop.results[1]
            with ir.InsertionPoint.after(loop):
                # The sum accumulator is in the softmax type; widen it to the f32
                # accumulator type before dividing the f32 P@V result.
                if reduction_type != compute_type:
                    l_i_out = arith.extf(
                        ir.VectorType.get([*batch_shape, wg_rows], compute_type),
                        l_i_out,
                    )
                # Normalize the output: output_final = pv_out / l_i_out
                output_normalized = normalize_output_by_sum(
                    pv_out, l_i_out, batch_shape, wg_rows, d_head, compute_type
                )
                # Narrow back to the type of the output op being replaced, if needed
                output_type = output_op.results[0].type
                if ir.VectorType(output_type).element_type != compute_type:
                    output_final = arith.truncf(output_type, output_normalized)
                else:
                    output_final = output_normalized

            # Replace all uses of the original output operation with the final loop result
            output_op.results[0].replace_all_uses_with(output_final)

            # Erase the original output operation
            rewriter.erase_op(output_op)

            # Return the final output handle
            results.set_ops(op.new_output, [output_final.owner])
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "ReplaceWithFusedAttentionOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: ir.Operation):
            return (
                # Read Q, K, scale, V slices
                transform.only_reads_handle(op.op_operands[:4])
                # Consume and replace output
                + transform.consumes_handle(op.op_operands[4:5])
                # Produce new output handle
                + transform.produces_handle(op.results)
                # Modify the payload
                + transform.modifies_payload()
            )


def replace_with_fused_attention(
    q_load: ir.Value,
    k_load: ir.Value,
    v_load: ir.Value,
    scale: ir.Value,
    output: ir.Value,
    tile_size: int | ir.IntegerAttr,
) -> ir.Value:
    """Replace a given (standard) attention output with an equivalent output
    that is computed in a fused fashion (fused attention optimization).

    Args:
        q_load: Handle to Q load operation (vector.transfer_read)
        k_load: Handle to K load operation (vector.transfer_read)
        v_load: Handle to V load operation (vector.transfer_read)
        scale: Handle to scale constant operation (arith.constant)
        output: Handle to output operation to replace (vector.contract)
        tile_size: Tile size for the reduction dimension tiling (K/V sequence length)

    Returns:
        Handle to the new output operation
    """
    if not isinstance(tile_size, ir.IntegerAttr):
        tile_size = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), tile_size)

    return ReplaceWithFusedAttentionOp(
        q_load, k_load, v_load, scale, output, tile_size=tile_size
    ).new_output

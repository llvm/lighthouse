"""Transform extension to generate fused attention computation."""

import numpy as np
from mlir import ir
from mlir.dialects import ext, transform, arith, scf, math, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def emit_vector_constant(shape, fill_value, element_type):
    """Emit an arith.constant of vector type, filled with fill_value."""
    vector_type = ir.VectorType.get(list(shape), element_type)
    np_dtype = np.float16 if element_type == ir.F16Type.get() else np.float32
    values = np.full(shape, fill_value, dtype=np_dtype)
    attr = ir.DenseElementsAttr.get(values, type=vector_type)
    return arith.constant(vector_type, attr)


def compute_qkt(
    q_value,
    k_load_op,
    loop_idx,
    wg_rows,
    d_head,
    tile_size,
    element_type,
    compute_type,
):
    """Load the K tile, transpose it, and contract with Q to produce Q@K^T.

    The K tile is [tile_size, d_head], transposed to [d_head, tile_size] and
    contracted with q_value [wg_rows, d_head] to produce [wg_rows, tile_size].
    Q and K are `element_type`, the contraction accumulates in `compute_type`.
    """
    k_memref = k_load_op.operands[0]
    k_load_indices = list(k_load_op.operands[1:-1])
    padding = k_load_op.operands[-1]
    in_bounds = k_load_op.attributes.get("in_bounds", None)
    k_perm_map = k_load_op.attributes.get("permutation_map", None)

    affine_d0 = ir.AffineExpr.get_dim(0)
    affine_d1 = ir.AffineExpr.get_dim(1)
    affine_d2 = ir.AffineExpr.get_dim(2)

    q_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d2])
    k_map = ir.AffineMap.get(3, 0, [affine_d2, affine_d1])
    out_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d1])

    indexing_maps = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(q_map),
            ir.AffineMapAttr.get(k_map),
            ir.AffineMapAttr.get(out_map),
        ]
    )

    iterator_types = ir.ArrayAttr.get(
        [
            ir.Attribute.parse("#vector.iterator_type<parallel>"),
            ir.Attribute.parse("#vector.iterator_type<parallel>"),
            ir.Attribute.parse("#vector.iterator_type<reduction>"),
        ]
    )

    qkt_type = ir.VectorType.get([wg_rows, tile_size], compute_type)
    qkt_acc = emit_vector_constant((wg_rows, tile_size), 0.0, compute_type)

    k_tile_indices = k_load_indices.copy()
    k_tile_indices[-2] = loop_idx

    k_tile_type = ir.VectorType.get([tile_size, d_head], element_type)
    k_tile = vector.TransferReadOp(
        k_tile_type,
        k_memref,
        k_tile_indices,
        k_perm_map,
        padding,
        in_bounds=in_bounds,
    ).result

    k_transpose_type = ir.VectorType.get([d_head, tile_size], element_type)
    k_transpose = vector.transpose(k_transpose_type, k_tile, [1, 0])

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
    wg_rows,
    tile_size,
    compute_type,
):
    """Apply online softmax to the scaled Q@K^T and reduce to a row-wise sum.

    Computes exp(qkt_scaled - m_ij), with m_ij broadcast over the inner dim.
    Returns (qkt_exp, l_ij) where qkt_exp is the [wg_rows, tile_size] exp tile
    and l_ij is its row-wise sum [wg_rows] (added into l_i_init).
    """
    # Broadcast m_ij from [wg_rows] to [wg_rows, tile_size]
    m_ij_bcasted_type = ir.VectorType.get([tile_size, wg_rows], compute_type)
    m_ij_bcasted = vector.broadcast(m_ij_bcasted_type, m_ij)
    m_ij_transposed_type = ir.VectorType.get([wg_rows, tile_size], compute_type)
    m_ij_transposed = vector.transpose(m_ij_transposed_type, m_ij_bcasted, [1, 0])

    qkt_centered = arith.subf(qkt_scaled, m_ij_transposed)
    # fastmath<fast> lets the exp lower to the native hardware exp; without it
    # the accurate expansion doubles the exp count and scalarizes part of it.
    qkt_exp = math.exp(qkt_centered, fastmath="fast")

    l_ij = vector.multi_reduction(
        kind="add",
        source=qkt_exp,
        acc=l_i_init,
        reduction_dims=[1],
    )

    return qkt_exp, l_ij


def rescale_pv_out_accumulator(acc, alpha, wg_rows, d_head, compute_type):
    """Rescale the running P@V accumulator by broadcasting alpha across d_head.

    Broadcasts alpha [wg_rows] to [wg_rows, d_head] and multiplies acc by it
    elementwise. Returns the rescaled accumulator.
    """
    alpha_bcasted_type = ir.VectorType.get([d_head, wg_rows], compute_type)
    alpha_bcasted = vector.broadcast(alpha_bcasted_type, alpha)
    alpha_transposed_type = ir.VectorType.get([wg_rows, d_head], compute_type)
    alpha_transposed = vector.transpose(alpha_transposed_type, alpha_bcasted, [1, 0])
    return arith.mulf(acc, alpha_transposed)


def compute_pv(
    qkt_exp,
    v_load_op,
    pv_init,
    loop_idx,
    acc_vector_type,
    d_head,
    tile_size,
    element_type,
):
    """Load the V tile and contract it with the softmax tile, accumulating into pv_init.

    Loads V [tile_size, d_head] (`element_type`) and contracts it with the exp
    tile [wg_rows, tile_size] into the running [wg_rows, d_head] accumulator,
    whose type is given by `acc_vector_type`. Returns the accumulated result.
    """
    v_memref = v_load_op.operands[0]
    v_load_indices = list(v_load_op.operands[1:-1])
    v_padding = v_load_op.operands[-1]
    v_in_bounds = v_load_op.attributes.get("in_bounds", None)
    v_perm_map = v_load_op.attributes.get("permutation_map", None)

    affine_d0 = ir.AffineExpr.get_dim(0)
    affine_d1 = ir.AffineExpr.get_dim(1)
    affine_d2 = ir.AffineExpr.get_dim(2)
    qkt_exp_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d2])
    v_map = ir.AffineMap.get(3, 0, [affine_d2, affine_d1])
    pv_out_map = ir.AffineMap.get(3, 0, [affine_d0, affine_d1])

    indexing_maps_pv = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(qkt_exp_map),
            ir.AffineMapAttr.get(v_map),
            ir.AffineMapAttr.get(pv_out_map),
        ]
    )

    iterator_types_pv = ir.ArrayAttr.get(
        [
            ir.Attribute.parse("#vector.iterator_type<parallel>"),
            ir.Attribute.parse("#vector.iterator_type<parallel>"),
            ir.Attribute.parse("#vector.iterator_type<reduction>"),
        ]
    )

    v_tile_indices = v_load_indices.copy()
    v_tile_indices[-2] = loop_idx

    v_tile_type = ir.VectorType.get([tile_size, d_head], element_type)
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


def normalize_ouput_by_sum(pv_out, l_i_out, wg_rows, d_head, compute_type):
    """Divide pv_out [wg_rows, d_head] by l_i_out [wg_rows] (broadcast over d_head)."""
    l_i_out_bcasted_type = ir.VectorType.get([d_head, wg_rows], compute_type)
    l_i_out_bcasted = vector.broadcast(l_i_out_bcasted_type, l_i_out)
    l_i_out_transposed_type = ir.VectorType.get([wg_rows, d_head], compute_type)
    l_i_out_transposed = vector.transpose(
        l_i_out_transposed_type, l_i_out_bcasted, [1, 0]
    )
    return arith.divf(pv_out, l_i_out_transposed)


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

            # Extract the scale scalar value from scale_op (arith.constant)
            scale_attr = scale_op.attributes["value"]
            scale_dense_attr = ir.DenseElementsAttr(scale_attr)
            scale_np_array = np.array(scale_dense_attr)
            scale_value = float(scale_np_array.flat[0])

            # Extract wg_rows and d_head from q_load result type
            q_load_result = q_load_op.results[0]
            q_vector_type = ir.VectorType(q_load_result.type)
            wg_rows = q_vector_type.shape[0]
            d_head = q_vector_type.shape[1]

            # Get tile size
            tile_size_value = ir.IntegerAttr(op.tile_size).value

            # Get element type from q_load result. Q, K, V and the output stay in this
            # type (the DPAS input type), while the softmax and both accumulators are
            # computed in f32 to keep the online softmax accurate and to avoid holding
            # the intermediates in narrow registers.
            element_type = q_vector_type.element_type
            compute_type = ir.F32Type.get()

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # Define m_i_init: vector of shape [wg_rows] with neg_inf values
                m_i_init = emit_vector_constant((wg_rows,), float("-inf"), compute_type)

                # Define l_i_init: vector of shape [wg_rows] with zero values
                l_i_init = emit_vector_constant((wg_rows,), 0.0, compute_type)

                # Define acc_init: vector of shape [wg_rows, d_head] with zero values
                acc_vector_type = ir.VectorType.get([wg_rows, d_head], compute_type)
                acc_init = emit_vector_constant((wg_rows, d_head), 0.0, compute_type)

                # Get n_ctx from k_load result type (first dimension size)
                k_load_result = k_load_op.results[0]
                k_vector_type = ir.VectorType(k_load_result.type)
                n_ctx = k_vector_type.shape[0]
                # Define scale tile: [wg_rows, tile_size] filled with the scale value
                scale_tile = emit_vector_constant(
                    (wg_rows, tile_size_value), scale_value, compute_type
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
                        wg_rows,
                        d_head,
                        tile_size_value,
                        element_type,
                        compute_type,
                    )
                    qkt_scaled = arith.mulf(qkt, scale_tile)

                    # Reduce the scaled Q@K^T to a row-wise max: [wg_rows]
                    qkt_row_max = vector.multi_reduction(
                        kind="maximumf",
                        source=qkt_scaled,
                        acc=m_i_init,
                        reduction_dims=[1],
                    )

                    # Compute m_ij = max(m_i, qkt_row_max)
                    # Both have shape [wg_rows]
                    m_ij = arith.maximumf(m_i, qkt_row_max)

                    # Apply online softmax and reduce to row-wise sum
                    qkt_exp, l_ij = compute_online_softmax_and_sum(
                        qkt_scaled,
                        m_ij,
                        l_i_init,
                        wg_rows,
                        tile_size_value,
                        compute_type,
                    )

                    # Compute alpha = exp(m_i - m_ij)
                    m_diff = arith.subf(m_i, m_ij)
                    alpha = math.exp(m_diff, fastmath="fast")

                    # Update l_i: l_i_updated = l_i * alpha + l_ij
                    l_i_scaled = arith.mulf(l_i, alpha)
                    l_i_updated = arith.addf(l_i_scaled, l_ij)

                    # Rescale running P@V accumulator by alpha
                    acc_updated = rescale_pv_out_accumulator(
                        acc, alpha, wg_rows, d_head, compute_type
                    )

                    # Narrow the softmax tile to the DPAS input type
                    qkt_exp_type = ir.VectorType.get(
                        [wg_rows, tile_size_value], element_type
                    )
                    qkt_exp_narrow = arith.truncf(qkt_exp_type, qkt_exp)

                    # Load the V tile and contract with the softmax tile into pv_out
                    pv_out = compute_pv(
                        qkt_exp_narrow,
                        v_load_op,
                        acc_updated,
                        loop_idx,
                        acc_vector_type,
                        d_head,
                        tile_size_value,
                        element_type,
                    )

                    # Yield the updated iter args
                    scf.yield_([m_ij, l_i_updated, pv_out])

            # Extract the final accumulator result (3rd output) from the loop
            pv_out = loop.results[2]
            l_i_out = loop.results[1]
            with ir.InsertionPoint.after(loop):
                # Normalize the output: output_final = pv_out / l_i_out
                output_normalized = normalize_ouput_by_sum(
                    pv_out, l_i_out, wg_rows, d_head, compute_type
                )
                # Narrow back to the type of the output op being replaced
                output_final = arith.truncf(
                    output_op.results[0].type, output_normalized
                )

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

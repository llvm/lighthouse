"""Transform extension to generate fused attention computation."""

import numpy as np
from mlir import ir
from mlir.dialects import ext, transform, arith, scf, math, vector
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


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

            # Get element type from q_load result
            element_type = q_vector_type.element_type

            # Build the fused attention computation
            with ir.InsertionPoint(output_op):
                # Define m_i_init: vector of shape [wg_rows] with neg_inf values
                # NOTE: We use float32 for the initial neg_inf values and cast to the element type
                # to avoid issues with representing -inf.
                m_i_vector_type = ir.VectorType.get([wg_rows], element_type)
                m_i_vector_type_f32 = ir.VectorType.get([wg_rows], ir.F32Type.get())
                neg_inf_value = float("-inf")
                m_i_values = np.full(
                    wg_rows,
                    neg_inf_value,
                    dtype=np.float32,
                )
                m_i_init_attr = ir.DenseElementsAttr.get(
                    m_i_values, type=m_i_vector_type_f32
                )
                m_i_init_f32 = arith.constant(m_i_vector_type_f32, m_i_init_attr)
                m_i_init = arith.truncf(m_i_vector_type, m_i_init_f32)

                # Define l_i_init: vector of shape [wg_rows] with zero values
                l_i_vector_type = ir.VectorType.get([wg_rows], element_type)
                l_i_values = np.zeros(
                    wg_rows,
                    dtype=np.float16
                    if element_type == ir.F16Type.get()
                    else np.float32,
                )
                l_i_init_attr = ir.DenseElementsAttr.get(
                    l_i_values, type=l_i_vector_type
                )
                l_i_init = arith.constant(l_i_vector_type, l_i_init_attr)

                # Define acc_init: vector of shape [wg_rows, d_head] with zero values
                acc_vector_type = ir.VectorType.get([wg_rows, d_head], element_type)
                acc_values = np.zeros(
                    (wg_rows, d_head),
                    dtype=np.float16
                    if element_type == ir.F16Type.get()
                    else np.float32,
                )
                acc_init_attr = ir.DenseElementsAttr.get(
                    acc_values, type=acc_vector_type
                )
                acc_init = arith.constant(acc_vector_type, acc_init_attr)

                # Get n_ctx from k_load result type (first dimension size)
                k_load_result = k_load_op.results[0]
                k_vector_type = ir.VectorType(k_load_result.type)
                n_ctx = k_vector_type.shape[0]
                # Define scale vector: vector of shape [wg_rows] with the scale value
                scale_vector_type = ir.VectorType.get([wg_rows], element_type)
                scale_values = np.full(
                    (wg_rows),
                    scale_value,
                    dtype=np.float16
                    if element_type == ir.F16Type.get()
                    else np.float32,
                )
                scale_init_attr = ir.DenseElementsAttr.get(
                    scale_values, type=scale_vector_type
                )
                scale_vector = arith.constant(scale_vector_type, scale_init_attr)

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

                    # Get common values for K/V tiling
                    k_memref = k_load_op.operands[0]
                    k_load_indices = list(k_load_op.operands[1:-1])
                    padding = k_load_op.operands[-1]
                    in_bounds = k_load_op.attributes.get("in_bounds", None)
                    k_perm_map = k_load_op.attributes.get("permutation_map", None)
                    q_value = q_load_op.results[0]

                    # Constants for K/V tiling (tile into chunks of 16)
                    k_subtile_size = 16
                    num_k_tiles = tile_size_value // k_subtile_size

                    # Create offset constants for each K tile
                    k_tile_offsets = []
                    for i in range(num_k_tiles):
                        offset = arith.constant(index_type, i * k_subtile_size)
                        k_tile_offsets.append(offset)

                    # Load and process K tiles (unrolled)
                    # Each K tile is [16, d_head], transposed to [d_head, 16], contracted to [wg_rows, 16]
                    qkt_chunks = []

                    # Create affine maps for Q@K contraction (used for all tiles)
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

                    # Accumulator for Q@K chunks
                    qkt_chunk_type = ir.VectorType.get(
                        [wg_rows, k_subtile_size], element_type
                    )
                    qkt_chunk_acc_values = np.zeros(
                        (wg_rows, k_subtile_size),
                        dtype=np.float16
                        if element_type == ir.F16Type.get()
                        else np.float32,
                    )
                    qkt_chunk_acc_attr = ir.DenseElementsAttr.get(
                        qkt_chunk_acc_values, type=qkt_chunk_type
                    )
                    qkt_chunk_acc = arith.constant(qkt_chunk_type, qkt_chunk_acc_attr)

                    for tile_idx in range(num_k_tiles):
                        # Compute the offset index for this tile
                        k_tile_offset = arith.addi(loop_idx, k_tile_offsets[tile_idx])

                        # Update indices for this K tile
                        k_tile_indices = k_load_indices.copy()
                        k_tile_indices[-2] = k_tile_offset

                        # Load K tile: [16, d_head]
                        k_tile_type = ir.VectorType.get(
                            [k_subtile_size, d_head], element_type
                        )
                        k_tile = vector.TransferReadOp(
                            k_tile_type,
                            k_memref,
                            k_tile_indices,
                            k_perm_map,
                            padding,
                            in_bounds=in_bounds,
                        ).result

                        # Transpose K tile: [16, d_head] -> [d_head, 16]
                        k_transpose_type = ir.VectorType.get(
                            [d_head, k_subtile_size], element_type
                        )
                        k_transpose = vector.transpose(k_transpose_type, k_tile, [1, 0])

                        # Contract Q @ K_transpose: [wg_rows, d_head] @ [d_head, 16] -> [wg_rows, 16]
                        qkt_chunk = vector.contract(
                            qkt_chunk_type,
                            q_value,
                            k_transpose,
                            qkt_chunk_acc,
                            indexing_maps=indexing_maps,
                            iterator_types=iterator_types,
                        )
                        qkt_chunks.append(qkt_chunk)

                    # Elementwise maximum across all Q@K chunks
                    # Build tree of maximumf operations
                    qkt_max_combined = qkt_chunks[0]
                    for i in range(1, num_k_tiles):
                        qkt_max_combined = arith.maximumf(
                            qkt_max_combined, qkt_chunks[i]
                        )

                    # Final multi_reduction to get row-wise max: [wg_rows, 16] -> [wg_rows]
                    qkt_max = vector.multi_reduction(
                        kind="maxnumf",
                        source=qkt_max_combined,
                        acc=m_i_init,
                        reduction_dims=[1],
                    )

                    # Scale the max: qkt_max_scaled = qkt_max * scale
                    # Both have shape [wg_rows]
                    qkt_max_scaled = arith.mulf(qkt_max, scale_vector)

                    # Compute m_ij = max(m_i, qkt_max_scaled)
                    # Both have shape [wg_rows]
                    m_ij = arith.maximumf(m_i, qkt_max_scaled)

                    # Apply softmax to each Q@K chunk
                    # Scale constant for chunks: [wg_rows, 16]
                    scale_chunk_type = ir.VectorType.get(
                        [wg_rows, k_subtile_size], element_type
                    )
                    scale_chunk_values = np.full(
                        (wg_rows, k_subtile_size),
                        scale_value,
                        dtype=np.float16
                        if element_type == ir.F16Type.get()
                        else np.float32,
                    )
                    scale_chunk_attr = ir.DenseElementsAttr.get(
                        scale_chunk_values, type=scale_chunk_type
                    )
                    scale_chunk = arith.constant(scale_chunk_type, scale_chunk_attr)

                    # Broadcast m_ij from [wg_rows] to [wg_rows, 16]
                    m_ij_bcasted_type = ir.VectorType.get(
                        [k_subtile_size, wg_rows], element_type
                    )
                    m_ij_bcasted = vector.broadcast(m_ij_bcasted_type, m_ij)
                    m_ij_transposed_type = ir.VectorType.get(
                        [wg_rows, k_subtile_size], element_type
                    )
                    m_ij_transposed = vector.transpose(
                        m_ij_transposed_type, m_ij_bcasted, [1, 0]
                    )

                    # Apply softmax to each chunk
                    qkt_exp_chunks = []
                    for qkt_chunk in qkt_chunks:
                        # Scale: qkt_scaled = qkt_chunk * scale
                        qkt_scaled = arith.mulf(qkt_chunk, scale_chunk)

                        # Center: qkt_centered = qkt_scaled - m_ij_transposed
                        qkt_centered = arith.subf(qkt_scaled, m_ij_transposed)

                        # Exponential: qkt_exp = exp(qkt_centered)
                        qkt_exp = math.exp(qkt_centered)
                        qkt_exp_chunks.append(qkt_exp)

                    # Elementwise sum across all exp chunks
                    qkt_exp_combined = qkt_exp_chunks[0]
                    for i in range(1, num_k_tiles):
                        qkt_exp_combined = arith.addf(
                            qkt_exp_combined, qkt_exp_chunks[i]
                        )

                    # Final multi_reduction to get row-wise sum: [wg_rows, 16] -> [wg_rows]
                    l_ij = vector.multi_reduction(
                        kind="add",
                        source=qkt_exp_combined,
                        acc=l_i_init,
                        reduction_dims=[1],
                    )

                    # Compute alpha = exp(m_i - m_ij)
                    m_diff = arith.subf(m_i, m_ij)
                    alpha = math.exp(m_diff)

                    # Update l_i: l_i_updated = l_i * alpha + l_ij
                    l_i_scaled = arith.mulf(l_i, alpha)
                    l_i_updated = arith.addf(l_i_scaled, l_ij)

                    # Broadcast alpha from [wg_rows] to [wg_rows, d_head]
                    alpha_bcasted_type = ir.VectorType.get(
                        [d_head, wg_rows], element_type
                    )
                    alpha_bcasted = vector.broadcast(alpha_bcasted_type, alpha)
                    alpha_transposed_type = ir.VectorType.get(
                        [wg_rows, d_head], element_type
                    )
                    alpha_transposed = vector.transpose(
                        alpha_transposed_type, alpha_bcasted, [1, 0]
                    )

                    # Update accumulator: acc_updated = acc * alpha_bcasted
                    acc_updated = arith.mulf(acc, alpha_transposed)

                    # Load V tiles and compute attention-weighted values
                    # Get V load parameters
                    v_memref = v_load_op.operands[0]
                    v_load_indices = list(v_load_op.operands[1:-1])
                    v_padding = v_load_op.operands[-1]
                    v_in_bounds = v_load_op.attributes.get("in_bounds", None)
                    v_perm_map = v_load_op.attributes.get("permutation_map", None)

                    # Create affine maps for P@V contraction
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

                    # Load and process V tiles (unrolled), accumulating results
                    pv_out = acc_updated
                    for tile_idx in range(num_k_tiles):
                        # Compute the offset index for this V tile
                        v_tile_offset = arith.addi(loop_idx, k_tile_offsets[tile_idx])

                        # Update indices for this V tile
                        v_tile_indices = v_load_indices.copy()
                        v_tile_indices[-2] = v_tile_offset

                        # Load V tile: [16, d_head]
                        v_tile_type = ir.VectorType.get(
                            [k_subtile_size, d_head], element_type
                        )
                        v_tile = vector.TransferReadOp(
                            v_tile_type,
                            v_memref,
                            v_tile_indices,
                            v_perm_map,
                            v_padding,
                            in_bounds=v_in_bounds,
                        ).result

                        # Contract qkt_exp_chunk @ v_tile: [wg_rows, 16] @ [16, d_head] -> [wg_rows, d_head]
                        # Accumulate into pv_out
                        pv_out = vector.contract(
                            acc_vector_type,
                            qkt_exp_chunks[tile_idx],
                            v_tile,
                            pv_out,
                            indexing_maps=indexing_maps_pv,
                            iterator_types=iterator_types_pv,
                        )

                    # Yield the updated iter args
                    scf.yield_([m_ij, l_i_updated, pv_out])

            # Extract the final accumulator result (3rd output) from the loop
            pv_out = loop.results[2]
            l_i_out = loop.results[1]
            with ir.InsertionPoint.after(loop):
                # Normalize the output: output_final = pv_out / l_i_out
                # Need to broadcast l_i_out from [wg_rows] to [wg_rows, d_head]
                l_i_out_bcasted_type = ir.VectorType.get(
                    [d_head, wg_rows], element_type
                )
                l_i_out_bcasted = vector.broadcast(l_i_out_bcasted_type, l_i_out)
                l_i_out_transposed_type = ir.VectorType.get(
                    [wg_rows, d_head], element_type
                )
                l_i_out_transposed = vector.transpose(
                    l_i_out_transposed_type, l_i_out_bcasted, [1, 0]
                )
                output_final = arith.divf(pv_out, l_i_out_transposed)

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
        def get_effects(op: ir.Operation, effects):
            # Read Q, K, scale, V slices
            transform.only_reads_handle(op.op_operands[:4], effects)
            # Consume and replace output
            transform.consumes_handle(op.op_operands[4:5], effects)
            # Produce new output handle
            transform.produces_handle(op.results, effects)
            # Modify the payload
            transform.modifies_payload(effects)


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

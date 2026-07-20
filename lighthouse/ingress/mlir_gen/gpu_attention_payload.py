"""Generate MLIR payload for GPU attention operation."""

import math

from mlir import ir
from mlir.dialects import (
    arith,
    bufferization,
    linalg,
    math as mlir_math,
    memref,
    tensor,
)

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import (
    affine_map,
    emit_buf_to_tensor,
    parallel,
    reduction,
)


def generate_gpu_attention_payload(
    func_name: str,
    Z: int,
    H: int,
    n_ctx: int,
    n_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for attention payload.

    Computes attention:
    output = softmax(Q @ K^T / sqrt(n_head)) @ V

    Args:
        func_name: Name of the payload function
        Z: Batch size
        H: Number of attention heads
        n_ctx: Context length (sequence length)
        n_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the attention payload function
    """
    mod = ir.Module.create()
    shape = (Z, H, n_ctx, n_head)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Collapse first 2 dimensions (Z, H) into a batch dimension
        # From (Z, H, n_ctx, n_head) to (Z*H, n_ctx, n_head)
        batch_dim = Z * H
        collapsed_shape_3d = (batch_dim, n_ctx, n_head)
        memref_3d_t = ir.MemRefType.get(collapsed_shape_3d, dtype)

        # Function signature: payload(output, Q, K, V)
        @func_cif(memref_t, memref_t, memref_t, memref_t, name=func_name)
        def payload(output, Q_arg, K_arg, V_arg):
            # Collapse memrefs from 4D to 3D
            Q_3d_memref = memref.collapse_shape(
                memref_3d_t,
                Q_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            K_3d_memref = memref.collapse_shape(
                memref_3d_t,
                K_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            V_3d_memref = memref.collapse_shape(
                memref_3d_t,
                V_arg,
                reassociation=[[0, 1], [2], [3]],
            )
            output_3d_memref = memref.collapse_shape(
                memref_3d_t,
                output,
                reassociation=[[0, 1], [2], [3]],
            )

            # Convert 3D memrefs to tensors
            Q_3d = emit_buf_to_tensor(Q_3d_memref, restrict=True)
            K_3d = emit_buf_to_tensor(K_3d_memref, restrict=True)
            V_3d = emit_buf_to_tensor(V_3d_memref, restrict=True)

            # Step 1: Transpose K to get K^T
            # Permute from (batch_dim, n_ctx, n_head) to (batch_dim, n_head, n_ctx)
            kt_shape_3d = (batch_dim, n_head, n_ctx)
            kt_init = tensor.empty(kt_shape_3d, dtype)
            K_transposed = linalg.transpose(K_3d, outs=[kt_init], permutation=[0, 2, 1])

            # Step 2: Compute Q @ K^T using batch_matmul
            # Q: (batch_dim, n_ctx, n_head) @ K^T: (batch_dim, n_head, n_ctx)
            # Result: (batch_dim, n_ctx, n_ctx)
            qkt_shape_3d = (batch_dim, n_ctx, n_ctx)
            qkt_init = tensor.empty(qkt_shape_3d, dtype)
            # Initialize with zeros for matmul accumulation
            zero = arith.constant(dtype, 0.0)
            qkt_init_filled = linalg.fill(zero, outs=[qkt_init])

            # Batch matmul: Q @ K^T
            qkt = linalg.batch_matmul(Q_3d, K_transposed, outs=[qkt_init_filled])

            # Step 3: Scale by 1/sqrt(n_head)
            scale_factor = 1.0 / math.sqrt(n_head)
            scale_const = arith.constant(dtype, scale_factor)

            # Create a tensor filled with the scale factor
            scale_tensor_init = tensor.empty(qkt_shape_3d, dtype)
            scale_tensor = linalg.fill(scale_const, outs=[scale_tensor_init])

            # Elementwise multiply qkt with scale tensor
            scaled_qkt_init = tensor.empty(qkt_shape_3d, dtype)
            scaled_qkt = linalg.mul(qkt, scale_tensor, outs=[scaled_qkt_init])

            # Step 4: Softmax numerator, emitted in decomposed form but WITHOUT
            # the final division. This is the flash-attention reordering: keep
            # the unnormalized probabilities P = exp(x - rowmax) and the per-row
            # sum, and defer the division until after the P*V matmul (the row-sum
            # normalization commutes past P*V since it is a per-row scalar).
            #
            # Softmax reduces over the last dimension (j, dim=2). The iteration
            # space is 3D (batch, i, j); reductions reduce over j.
            reduce_shape_3d = (batch_dim, n_ctx)
            d0 = ir.AffineDimExpr.get(0)
            d1 = ir.AffineDimExpr.get(1)
            d2 = ir.AffineDimExpr.get(2)
            par_map_3d = affine_map(3, [d0, d1, d2])
            red_map_3d = affine_map(3, [d0, d1])

            # Step 4a: Row max reduction: max[b, i] = max_j scaled_qkt[b, i, j].
            neg_inf = arith.constant(dtype, float("-inf"))
            max_init = tensor.empty(reduce_shape_3d, dtype)
            max_acc = linalg.fill(neg_inf, outs=[max_init])

            @linalg.generic(
                [scaled_qkt],
                [max_acc],
                [par_map_3d, red_map_3d],
                [parallel, parallel, reduction],
            )
            def row_max(x, acc):
                return arith.MaximumFOp(x, acc)

            # Step 4b: Numerator P = exp(x - rowmax).
            p_init = tensor.empty(qkt_shape_3d, dtype)

            @linalg.generic(
                [scaled_qkt, row_max],
                [p_init],
                [par_map_3d, red_map_3d, par_map_3d],
                [parallel, parallel, parallel],
            )
            def numerator(x, m, _out):
                centered = arith.SubFOp(x, m).result
                return mlir_math.exp(centered)

            # Step 4c: Row sum reduction over the numerator:
            #    sum[b, i] = sum_j P[b, i, j].
            sum_init = tensor.empty(reduce_shape_3d, dtype)
            sum_acc = linalg.fill(zero, outs=[sum_init])

            @linalg.generic(
                [numerator],
                [sum_acc],
                [par_map_3d, red_map_3d],
                [parallel, parallel, reduction],
            )
            def row_sum(p, acc):
                return arith.AddFOp(p, acc)

            # Step 5: Multiply the UNNORMALIZED numerator P by V using batch_matmul.
            # P: (batch_dim, n_ctx, n_ctx) @ V: (batch_dim, n_ctx, n_head)
            # Result: (batch_dim, n_ctx, n_head)
            pv_init = tensor.empty(collapsed_shape_3d, dtype)
            pv_init_filled = linalg.fill(zero, outs=[pv_init])

            pv = linalg.batch_matmul(numerator, V_3d, outs=[pv_init_filled])

            # Step 6: Apply the deferred division by the row sum:
            #    out[b, i, k] = pv[b, i, k] / sum[b, i].
            result_3d_init = tensor.empty(collapsed_shape_3d, dtype)

            @linalg.generic(
                [pv, row_sum],
                [result_3d_init],
                [par_map_3d, red_map_3d, par_map_3d],
                [parallel, parallel, parallel],
            )
            def result_3d(pv_val, s, _out):
                return arith.DivFOp(pv_val, s)

            # Materialize 3D result back to 3D output memref
            bufferization.materialize_in_destination(
                None, result_3d, output_3d_memref, restrict=True, writable=True
            )

    return mod

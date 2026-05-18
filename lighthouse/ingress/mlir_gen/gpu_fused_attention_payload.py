"""Generate MLIR payload for GPU fused attention operation."""

import math

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, memref, tensor

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor


def generate_gpu_fused_attention_payload(
    func_name: str,
    Z: int,
    H: int,
    n_ctx: int,
    n_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for fused attention payload.

    Computes fused attention:
    output = softmax(Q @ K^T / sqrt(n_head)) @ V

    Args:
        func_name: Name of the payload function
        Z: Batch size
        H: Number of attention heads
        n_ctx: Context length (sequence length)
        n_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the fused attention payload function
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

            # Step 4: Apply softmax along the last dimension (dim=2 in 3D)
            softmax_init = tensor.empty(qkt_shape_3d, dtype)
            attention_weights = linalg.softmax(
                result=[ir.RankedTensorType.get(qkt_shape_3d, dtype)],
                input=scaled_qkt,
                output=softmax_init,
                dimension=2,
            )

            # Step 5: Multiply attention weights by V using batch_matmul
            # attention_weights: (batch_dim, n_ctx, n_ctx) @ V: (batch_dim, n_ctx, n_head)
            # Result: (batch_dim, n_ctx, n_head)
            output_3d_init = tensor.empty(collapsed_shape_3d, dtype)
            output_3d_init_filled = linalg.fill(zero, outs=[output_3d_init])

            result_3d = linalg.batch_matmul(
                attention_weights, V_3d, outs=[output_3d_init_filled]
            )

            # Materialize 3D result back to 3D output memref
            bufferization.materialize_in_destination(
                None, result_3d, output_3d_memref, restrict=True, writable=True
            )

        # Emit utility functions for GPU memory management
        emit_gpu_util_funcs(dtype, rank=4)

    return mod

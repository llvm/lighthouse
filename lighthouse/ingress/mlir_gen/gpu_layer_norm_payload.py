"""Generate MLIR payload for GPU layer_norm operation."""

from mlir import ir
from mlir.dialects import linalg, bufferization, tensor, arith, math

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import (
    emit_buf_to_tensor,
    affine_map,
    parallel,
    reduction,
)


def emit_layer_norm_generics(
    input_tensor, gamma_tensor, beta_tensor, out_init, N, dtype, eps=1e-5
):
    """Emit the 3 linalg.generic ops of a row-wise layer_norm and return the
    normalized result tensor.

    Computes, over iteration space (i, j) reducing on j:
        mean_i    = (1/N) * sum_j x[i, j]
        var_i     = (1/N) * sum_j (x[i, j] - mean_i)^2
        out[i, j] = (x[i, j] - mean_i) * rsqrt(var_i + eps) * gamma[j] + beta[j]

    `out_init` is the destination tensor for the final (normalize) generic; the
    two reduction accumulators are allocated internally and zero-filled. This is
    the payload core shared by the standalone gpu_layer_norm payload and the
    nanoGPT combined payload.
    """
    par_map_2d = affine_map(2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
    red_map_2d = affine_map(2, [ir.AffineDimExpr.get(0)])
    bias_map_2d = affine_map(2, [ir.AffineDimExpr.get(1)])

    zero = arith.constant(dtype, 0.0)
    inv_n_const = arith.constant(dtype, 1.0 / float(N))
    eps_const = arith.constant(dtype, eps)

    # Reduction accumulators are rank-1 (M,), inferred from the destination row count.
    reduce_shape = (ir.RankedTensorType(out_init.type).shape[0],)

    # 1) Mean reduction: mean_sum[i] = sum_j x[i, j]
    mean_acc = linalg.fill(zero, outs=[tensor.empty(reduce_shape, dtype)])

    @linalg.generic(
        [input_tensor],
        [mean_acc],
        [par_map_2d, red_map_2d],
        [parallel, reduction],
    )
    def mean_sum(x, acc):
        return arith.AddFOp(x, acc)

    # 2) Variance reduction: var_sum[i] = sum_j (x[i, j] - mean_i)^2
    var_acc = linalg.fill(zero, outs=[tensor.empty(reduce_shape, dtype)])

    @linalg.generic(
        [input_tensor, mean_sum],
        [var_acc],
        [par_map_2d, red_map_2d, red_map_2d],
        [parallel, reduction],
    )
    def var_sum(x, m_sum, acc):
        mean = arith.MulFOp(m_sum, inv_n_const).result
        centered = arith.SubFOp(x, mean).result
        sq = arith.MulFOp(centered, centered).result
        return arith.AddFOp(sq, acc)

    # 3) Final elementwise: out[i, j] = (x[i, j] - mean_i) * rsqrt(var_i + eps)
    #                                   * gamma[j] + beta[j]
    @linalg.generic(
        [input_tensor, mean_sum, var_sum, gamma_tensor, beta_tensor],
        [out_init],
        [par_map_2d, red_map_2d, red_map_2d, bias_map_2d, bias_map_2d, par_map_2d],
        [parallel, parallel],
    )
    def normalized(x, m_sum, v_sum, g, b, _out):
        mean = arith.MulFOp(m_sum, inv_n_const).result
        var = arith.MulFOp(v_sum, inv_n_const).result
        var_eps = arith.AddFOp(var, eps_const).result
        inv_std = math.rsqrt(var_eps)
        centered = arith.SubFOp(x, mean).result
        scaled = arith.MulFOp(centered, inv_std).result
        weighted = arith.MulFOp(scaled, g).result
        return arith.AddFOp(weighted, b)

    return normalized


def generate_gpu_layer_norm_payload(
    func_name: str,
    M: int,
    N: int,
    dtype: ir.Type,
    eps: float = 1e-5,
) -> ir.Module:
    """
    Generate MLIR module for layer_norm payload.

    Computes layer normalization along the last dimension (rows):
        mean_i    = (1/N) * sum_j x[i, j]
        var_i     = (1/N) * sum_j (x[i, j] - mean_i)^2
        out[i, j] = (x[i, j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]

    Args:
        func_name: Name of the payload function
        M: Number of rows
        N: Number of columns (normalization dimension)
        dtype: MLIR element type (e.g., F32Type)
        eps: Small constant added to variance for numerical stability

    Returns:
        MLIR module containing the layer_norm payload function
    """
    mod = ir.Module.create()
    shape = (M, N)
    bias_shape = (N,)
    memref_t = ir.MemRefType.get(shape, dtype)
    bias_memref_t = ir.MemRefType.get(bias_shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, input, gamma, beta)
        @func_cif(memref_t, memref_t, bias_memref_t, bias_memref_t, name=func_name)
        def payload(output, input_arg, gamma_arg, beta_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input_arg, restrict=True)
            gamma_tensor = emit_buf_to_tensor(gamma_arg, restrict=True)
            beta_tensor = emit_buf_to_tensor(beta_arg, restrict=True)

            normalized = emit_layer_norm_generics(
                input_tensor,
                gamma_tensor,
                beta_tensor,
                tensor.empty(shape, dtype),
                N,
                dtype,
                eps,
            )
            bufferization.materialize_in_destination(
                None, normalized, output, restrict=True, writable=True
            )

    return mod

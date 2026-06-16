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
    reduce_shape = (M,)
    bias_shape = (N,)
    memref_t = ir.MemRefType.get(shape, dtype)
    bias_memref_t = ir.MemRefType.get(bias_shape, dtype)

    # Affine maps used by the linalg.generic ops below.
    # Iteration space is (i, j); reductions reduce over j.
    par_map_2d = affine_map(2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
    red_map_2d = affine_map(2, [ir.AffineDimExpr.get(0)])
    bias_map_2d = affine_map(2, [ir.AffineDimExpr.get(1)])

    inv_N = 1.0 / float(N)

    with ir.InsertionPoint(mod.body):
        # Function signature: payload(output, input, gamma, beta)
        @func_cif(memref_t, memref_t, bias_memref_t, bias_memref_t, name=func_name)
        def payload(output, input_arg, gamma_arg, beta_arg):
            emit_buf_to_tensor(output, restrict=True, writable=True)
            input_tensor = emit_buf_to_tensor(input_arg, restrict=True)
            gamma_tensor = emit_buf_to_tensor(gamma_arg, restrict=True)
            beta_tensor = emit_buf_to_tensor(beta_arg, restrict=True)

            zero = arith.constant(dtype, 0.0)
            inv_n_const = arith.constant(dtype, inv_N)
            eps_const = arith.constant(dtype, eps)

            # 1) Mean reduction: mean_sum[i, 0] = sum_j x[i, j]
            mean_init = tensor.empty(reduce_shape, dtype)
            mean_acc = linalg.fill(zero, outs=[mean_init])

            @linalg.generic(
                [input_tensor],
                [mean_acc],
                [par_map_2d, red_map_2d],
                [parallel, reduction],
            )
            def mean_sum(x, acc):
                return arith.AddFOp(x, acc)

            # 2) Variance reduction: var_sum[i, 0] = sum_j (x[i, j] - mean_i)^2
            #    where mean_i = mean_sum[i, 0] * (1/N)
            var_init = tensor.empty(reduce_shape, dtype)
            var_acc = linalg.fill(zero, outs=[var_init])

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

            # 3) Final elementwise:
            #    out[i, j] = (x[i, j] - mean_i) * rsqrt(var_i + eps) * gamma[j] + beta[j]
            out_init = tensor.empty(shape, dtype)

            @linalg.generic(
                [input_tensor, mean_sum, var_sum, gamma_tensor, beta_tensor],
                [out_init],
                [
                    par_map_2d,
                    red_map_2d,
                    red_map_2d,
                    bias_map_2d,
                    bias_map_2d,
                    par_map_2d,
                ],
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

            bufferization.materialize_in_destination(
                None, normalized, output, restrict=True, writable=True
            )

    return mod

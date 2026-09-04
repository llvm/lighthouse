"""Generate MLIR payload for GPU attention operation."""

import math

from mlir import ir
from mlir.dialects import arith, bufferization, linalg, memref, tensor
from mlir.dialects import math as math_dialect

from lighthouse.utils.mlir import func_cif
from lighthouse.ingress.mlir_gen.utils import emit_buf_to_tensor


def _element_type(value: ir.Value) -> ir.Type:
    """`value`'s element type if it is shaped, else its own type."""
    try:
        return ir.ShapedType(value.type).element_type
    except (ValueError, TypeError):
        return value.type


def _generic(ins, out, indexing_maps, iterator_types, body):
    """Emit a single-result ``linalg.generic`` over statically shaped tensors.

    `body` receives one scalar block argument per input followed by the output's,
    and returns the value to yield. Block argument types are taken per operand, so
    the operands and the result may differ in element type (a mixed-precision
    body).
    """
    maps = ir.ArrayAttr.get([ir.AffineMapAttr.get(m) for m in indexing_maps])
    iterators = ir.ArrayAttr.get(
        [ir.Attribute.parse(f"#linalg.iterator_type<{it}>") for it in iterator_types]
    )
    op = linalg.GenericOp(
        result_tensors=[out.type],
        inputs=ins,
        outputs=[out],
        indexing_maps=maps,
        iterator_types=iterators,
    )
    arg_types = [_element_type(v) for v in (*ins, out)]
    block = op.regions[0].blocks.append(*arg_types)
    with ir.InsertionPoint(block):
        linalg.yield_([body(*block.arguments)])
    return op.results[0]


def generate_gpu_attention_payload(
    func_name: str,
    batch_size: int,
    n_head: int,
    n_ctx: int,
    d_head: int,
    dtype: ir.Type,
) -> ir.Module:
    """
    Generate MLIR module for attention payload.

    Computes attention:
    output = softmax(Q @ K^T / sqrt(d_head)) @ V

    The softmax is emitted in the flash-attention form -- ``max``, ``exp``, row
    ``sum`` and the ``P@V`` contraction as separate ops, with the normalizing
    divide *after* the contraction -- rather than as a `linalg.softmax`. See
    step 4 for why.

    Args:
        func_name: Name of the payload function
        batch_size: Batch size
        n_head: Number of attention heads
        n_ctx: Context length (sequence length)
        d_head: Head dimension
        dtype: MLIR element type (e.g., F32Type)

    Returns:
        MLIR module containing the attention payload function
    """
    mod = ir.Module.create()
    shape = (batch_size, n_head, n_ctx, d_head)
    memref_t = ir.MemRefType.get(shape, dtype)

    with ir.InsertionPoint(mod.body):
        # Collapse first 2 dimensions (batch_size, n_head) into a batch dimension
        # From (batch_size, n_head, n_ctx, d_head) to (batch_size*n_head, n_ctx, d_head)
        batch_dim = batch_size * n_head
        collapsed_shape_3d = (batch_dim, n_ctx, d_head)
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
            # Permute from (batch_dim, n_ctx, d_head) to (batch_dim, d_head, n_ctx)
            kt_shape_3d = (batch_dim, d_head, n_ctx)
            kt_init = tensor.empty(kt_shape_3d, dtype)
            K_transposed = linalg.transpose(K_3d, outs=[kt_init], permutation=[0, 2, 1])

            # Step 2: Compute Q @ K^T using batch_matmul
            # Q: (batch_dim, n_ctx, d_head) @ K^T: (batch_dim, d_head, n_ctx)
            # Result: (batch_dim, n_ctx, n_ctx)
            # Mixed precision: both contractions keep narrow operands (f16 feeds the
            # DPAS units) but accumulate in f32, and the softmax runs in f32. The
            # extensions the mixed-type contraction puts in its body are folded back
            # into the `vector.contract` at vectorization by
            # `fold_type_extensions_into_contract`.
            compute_type = ir.F32Type.get()
            narrow = dtype != compute_type

            qkt_shape_3d = (batch_dim, n_ctx, n_ctx)
            qkt_init = tensor.empty(qkt_shape_3d, compute_type)
            # Initialize with zeros for matmul accumulation
            zero = arith.constant(compute_type, 0.0)
            qkt_init_filled = linalg.fill(zero, outs=[qkt_init])

            # Batch matmul: Q @ K^T, f16 operands accumulating in f32.
            qkt = linalg.batch_matmul(Q_3d, K_transposed, outs=[qkt_init_filled])

            # Step 3: Scale by 1/sqrt(d_head)
            scale_factor = 1.0 / math.sqrt(d_head)
            scale_const = arith.constant(compute_type, scale_factor)

            # Create a tensor filled with the scale factor
            scale_tensor_init = tensor.empty(qkt_shape_3d, compute_type)
            scale_tensor = linalg.fill(scale_const, outs=[scale_tensor_init])

            # Elementwise multiply qkt with scale tensor
            scaled_qkt_init = tensor.empty(qkt_shape_3d, compute_type)
            scaled_qkt = linalg.mul(qkt, scale_tensor, outs=[scaled_qkt_init])

            # Step 4: softmax over the last dimension, written in the
            # flash-attention form -- with the normalizing divide deferred past the
            # P@V contraction:
            #
            #   m   = max_k s          l   = sum_k P
            #   P   = exp(s - m)       O   = P @ V         out = O / l
            #
            # Algebraically identical to `softmax(s) @ V`: dividing by the per-row
            # `l` commutes with a contraction that reduces the *other* axis. Written
            # this way rather than as a `linalg.softmax` -- whose decomposition
            # normalizes *before* the contraction -- it leaves the
            # `max -> exp -> {sum, P@V}` chain explicit, which is what
            # `transform_ext.fuse_dependant_reduction_ops` folds into one loop.
            d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
            # (batch, row, col) -> (batch, row, col) and -> (batch, row): the
            # per-row statistics are broadcast over the reduced axis.
            elementwise_map = ir.AffineMap.get(3, 0, [d0, d1, d2])
            row_map = ir.AffineMap.get(3, 0, [d0, d1])
            row_shape_3d = (batch_dim, n_ctx)

            # m = max_k s
            neg_inf = arith.constant(compute_type, float("-inf"))
            m_init = linalg.fill(
                neg_inf, outs=[tensor.empty(row_shape_3d, compute_type)]
            )
            row_max = _generic(
                [scaled_qkt],
                m_init,
                [elementwise_map, row_map],
                ["parallel", "parallel", "reduction"],
                lambda s, acc: arith.maximumf(s, acc),
            )

            # P = exp(s - m), kept in f32 and read directly by both consumers: a
            # cast in between would hide the chain from the fusion.
            probs = _generic(
                [scaled_qkt, row_max],
                tensor.empty(qkt_shape_3d, compute_type),
                [elementwise_map, row_map, elementwise_map],
                ["parallel", "parallel", "parallel"],
                lambda s, m, out: math_dialect.exp(arith.subf(s, m)),
            )

            # l = sum_k P, all in f32.
            l_init = linalg.fill(zero, outs=[tensor.empty(row_shape_3d, compute_type)])
            row_sum = _generic(
                [probs],
                l_init,
                [elementwise_map, row_map],
                ["parallel", "parallel", "reduction"],
                lambda p, acc: arith.addf(p, acc),
            )

            # Step 5: O = P @ V, still unnormalized. A `linalg.generic` rather than a
            # `linalg.batch_matmul` so the narrowing of P can live *inside* the body:
            # trunc P, widen it and V back to f32, multiply, accumulate. The two
            # widenings are what `fold_type_extensions_into_contract` matches -- it
            # folds both into the `vector.contract`, leaving a narrow x narrow -> f32
            # contraction (the DPAS shape) with just the `truncf` outside.
            #
            # A named matmul cannot express this: it casts every operand to the
            # accumulator type, so an f32 P leaves nothing to fold on the lhs and V
            # gets widened instead, losing the f16 DPAS. Keeping the cast in the body
            # also keeps P readable straight from the elementwise term, so the fusion
            # still sees the chain.
            b, m, n, k = (ir.AffineDimExpr.get(i) for i in range(4))
            pv_lhs_map = ir.AffineMap.get(4, 0, [b, m, k])
            pv_rhs_map = ir.AffineMap.get(4, 0, [b, k, n])
            pv_out_map = ir.AffineMap.get(4, 0, [b, m, n])

            def contract(p, v, acc):
                lhs, rhs = p, v
                if narrow:
                    lhs = arith.extf(compute_type, arith.truncf(dtype, p))
                    rhs = arith.extf(compute_type, v)
                return arith.addf(acc, arith.mulf(lhs, rhs))

            output_3d_init = tensor.empty(collapsed_shape_3d, compute_type)
            output_3d_init_filled = linalg.fill(zero, outs=[output_3d_init])
            unnormalized = _generic(
                [probs, V_3d],
                output_3d_init_filled,
                [pv_lhs_map, pv_rhs_map, pv_out_map],
                ["parallel", "parallel", "parallel", "reduction"],
                contract,
            )

            # Step 6: out = O / l, the deferred normalization, narrowed back to the
            # payload's element type. `l` is broadcast over d_head, the contraction's
            # free axis.
            def normalize(o, denom, out):
                normalized = arith.divf(o, denom)
                if narrow:
                    return arith.truncf(dtype, normalized)
                return normalized

            result_3d = _generic(
                [unnormalized, row_sum],
                tensor.empty(collapsed_shape_3d, dtype),
                [elementwise_map, row_map, elementwise_map],
                ["parallel", "parallel", "parallel"],
                normalize,
            )

            # Materialize 3D result back to 3D output memref
            bufferization.materialize_in_destination(
                None, result_3d, output_3d_memref, restrict=True, writable=True
            )

    return mod

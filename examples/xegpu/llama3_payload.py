"""Generate the Llama-3 forward pass payload at the linalg level (GPU / XeGPU).

This is stage 1 -- the payload ("what to compute"): an MLIR module describing
the Llama forward at linalg level. Llama's building blocks:

  * RMSNorm
  * SwiGLU FFN     w2( silu(x @ w1) * (x @ w3) )   -- 3 matmuls, no bias
  * grouped-query attention via the fused flash kernel (causal by default),
    with RoPE applied to q/k
  * no biases anywhere (Llama uses bias=False on every Linear)

  -> class `Builder` (emits one op at a time) and `build_llama_payload`.

Each transformer block is
    h   = x + attn_proj( MHA( rms_attn(x) ) )       # attention sublayer
    out = h + swiglu_ffn( rms_ffn(h) )              # MLP sublayer
and the full model is
    x = tok_embeddings(tokens)         # embeddings done host-side
    for _ in range(n_layers): x = Block(x)
    x = rms_final(x); logits = x @ output_weight

Kernel-friendly dims (functional correctness, shapes are free): dim=256,
n_heads=4 (head_dim=64), hidden=1024, vocab=256.
"""

from mlir import ir
from mlir.dialects import linalg, bufferization, tensor, arith, math, gpu, memref

from lighthouse.ingress.mlir_gen.utils import (
    emit_buf_to_tensor,
    affine_map,
    parallel,
    reduction,
)
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.named import times_weights
from lighthouse.utils.mlir import func_cif


def F32():  # 32-bit float (used for accumulation / norms)
    return ir.F32Type.get()


def F16():  # 16-bit float (required by the GPU matmul units)
    return ir.F16Type.get()


# =============================================================================
# Payload: describe what to compute (high-level linalg ops; no tiling/XeGPU yet)
# =============================================================================
# Each Builder method emits one high-level op that writes its result into a fresh
# on-device buffer (`gpu.alloc`), and returns a tensor "view" of that buffer for
# the next op to read. Because each op writes a distinct device buffer, each will
# become its OWN GPU kernel later; the buffers are the on-device handoff between
# kernels (kernel N writes buffer B, kernel N+1 reads B -- no host round-trip).
#
# dtype convention: the GPU matmul (DPAS) hardware needs f16 inputs and produces
# an f32 result. RMSNorm/softmax run in f32. So between a norm/softmax and a
# matmul we insert an explicit f32->f16 `cast` op (its own kernel).
# =============================================================================
class Builder:
    """Emits the model's ops and remembers the order/kind of each one.

    `kinds` is the crucial bookkeeping: an ordered list, one entry per op emitted,
    recording its "class" so the schedule (stage 2) can later tile and annotate
    each kernel correctly. Classes:
      'mm'  = matmul (linalg.matmul)          -> DPAS systolic-array kernel
      'rms' = RMSNorm (2 generics + 1 fill)   -> reduction kernel (uses shared mem)
      'fa'  = flash multi-head attention -> one kernel (QK^T->softmax->@V,
              online-softmax over K/V tiles; causal mask added by the schedule).
      'ew'  = elementwise (cast / silu / mul / residual) -> row-parallel kernel
    The op build order in the payload == the order of `kinds` == the order the
    kernels appear in the final module, which is how the schedule matches them up.
    """

    def __init__(self, T):
        self.T = T
        self.f32, self.f16 = F32(), F16()
        self.kinds = []  # ordered kernel classes (see docstring)
        self.mm_shapes = []  # (M,N,K) per matmul, in build order (for per-mm params)
        self.to_dealloc = []  # device buffers to gpu.dealloc at the end

    def _buf(self, shape, dtype):
        # Allocate a DEVICE buffer (lives in GPU memory). Returns the memref.
        b = gpu.alloc(ir.MemRefType.get(shape, dtype), None, [], [], [])
        self.to_dealloc.append(b)
        return b

    def _par(self, rank=2):
        # Identity affine map (d0,d1,...) -> (d0,d1,...): a plain elementwise
        # access pattern where output[i,j] depends on input[i,j].
        return affine_map(rank, [ir.AffineDimExpr.get(i) for i in range(rank)])

    # ---- matmul: a(M,K) f16 @ b(K,N) f16 -> (M,N) f32 buffer ----
    def matmul(self, a, b, M, N, out_buf=None):
        # Standard C = A @ B. `times_weights` emits linalg.matmul; we first fill the
        # accumulator with 0. f16 inputs, f32 output -- matches the DPAS hardware.
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        acc = linalg.fill(arith.constant(self.f32, 0.0), outs=[out_t])
        res = times_weights(a, b, acc)
        bufferization.materialize_in_destination(
            None, res, buf, restrict=True, writable=True
        )
        self.kinds.append("mm")
        K = a.type.shape[-1]
        self.mm_shapes.append((M, N, K))
        if out_buf is not None:  # caller gave the final output buffer
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- RMSNorm(x (M,N) f32, weight (N,)) -> (M,N) f32 buffer ----
    def rmsnorm(self, x, weight, M, N, eps=1e-5):
        # RMSNorm: out[i,j] = x[i,j] * rsqrt(mean_k x[i,k]^2 + eps) * weight[j].
        # No mean-subtraction (unlike LayerNorm). Built from 2 linalg.generic ops:
        #   (1) ss[i]     = sum_k x[i,k]^2                  (row reduction)
        #   (2) out[i,j]  = x[i,j] * rsqrt(ss_i/N + eps) * weight[j]
        # Affine maps:
        #   par2  (d0,d1)->(d0,d1) : full 2-D elementwise
        #   red2  (d0,d1)->(d0)    : reduce over j -> one value per row
        #   bias2 (d0,d1)->(d1)    : weight indexed by column only
        f32 = self.f32
        par2, red2 = self._par(), affine_map(2, [ir.AffineDimExpr.get(0)])
        bias2 = affine_map(2, [ir.AffineDimExpr.get(1)])
        inv_n = arith.constant(f32, 1.0 / float(N))
        eps_c = arith.constant(f32, eps)
        zero = arith.constant(f32, 0.0)
        # (1) sum of squares -> ss (linalg.fill zeroes the accumulator first)
        ss_acc = linalg.fill(zero, outs=[tensor.empty((M,), f32)])

        @linalg.generic([x], [ss_acc], [par2, red2], [parallel, reduction])
        def ss_sum(v, acc):
            return arith.AddFOp(arith.MulFOp(v, v).result, acc)

        # (2) normalize + scale -> output
        buf = self._buf((M, N), f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic(
            [x, ss_sum, weight],
            [out_t],
            [par2, red2, bias2, par2],
            [parallel, parallel],
        )
        def normed(v, s, w, _o):
            ms = arith.MulFOp(s, inv_n).result
            inv_rms = math.rsqrt(arith.AddFOp(ms, eps_c).result)
            return arith.MulFOp(arith.MulFOp(v, inv_rms).result, w)

        bufferization.materialize_in_destination(
            None, normed, buf, restrict=True, writable=True
        )
        self.kinds.append("rms")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- elementwise cast f32 -> f16 ----
    def cast_f16(self, x, M, N):
        par2 = self._par()
        buf = self._buf((M, N), self.f16)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([x], [out_t], [par2, par2], [parallel, parallel])
        def c(s, _o):
            return arith.TruncFOp(self.f16, s)

        bufferization.materialize_in_destination(
            None, c, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- silu (swish): out = x * sigmoid(x)  (x (M,N) f32) ----
    def silu(self, x, M, N):
        par2 = self._par()
        one = arith.constant(self.f32, 1.0)
        buf = self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([x], [out_t], [par2, par2], [parallel, parallel])
        def s(v, _o):
            # sigmoid(v) = 1 / (1 + exp(-v)); silu = v * sigmoid(v)
            neg = arith.NegFOp(v).result
            sig = arith.DivFOp(one, arith.AddFOp(one, math.exp(neg)).result).result
            return arith.MulFOp(v, sig)

        bufferization.materialize_in_destination(
            None, s, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- elementwise multiply: out = a * b  (both (M,N) f32) ----
    def mul(self, a, b, M, N):
        par2 = self._par()
        buf = self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([a, b], [out_t], [par2, par2, par2], [parallel, parallel])
        def m(x, y, _o):
            return arith.MulFOp(x, y)

        bufferization.materialize_in_destination(
            None, m, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- residual add: out = a + b  (both (M,N) f32) ----
    def add(self, a, b, M, N, out_buf=None):
        par2 = self._par()
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([a, b], [out_t], [par2, par2, par2], [parallel, parallel])
        def r(x, y, _o):
            return arith.AddFOp(x, y)

        bufferization.materialize_in_destination(
            None, r, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        if out_buf is not None:
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- RoPE (rotary position embedding), half-split -> (T,D) f32 buffer ----
    def rope(self, src_buf, cos, sin, T, D, nh):
        # Rotary embedding on a (T, D=nh*hs) f32 projection buffer, applied per head.
        # HALF-SPLIT (GPT-NeoX / HF-Llama) convention: within each head's hs coords,
        # coordinate d in the first half pairs with d+hs/2 in the second half, and
        #   out[d]        = x[d]*cos[t,d] - x[d+half]*sin[t,d]
        #   out[d+half]   = x[d+half]*cos[t,d] + x[d]*sin[t,d]
        # One multi-output row-parallel linalg.generic. The head-dim halves are
        # CONTIGUOUS sub-blocks (no stride-2 gather, unlike the interleaved
        # convention). CRUCIAL: the buffers are viewed HEAD-OUTERMOST (nh, T, hs)
        # (the same strided transpose as heads_view), NOT (T, nh, hs). Under the
        # (1, wg_rows, 0) tiling each grid block then owns one head's (wg_rows, half)
        # 2D slab, which lowers to a BLOCK load_nd/store_nd. The (T, nh, hs) layout
        # instead makes the head a middle vector dim over a big stride, which
        # convert-vector-to-xegpu turns into a scatter/gather that crashes codegen.
        # cos/sin are (T, half) f32, indexed (row t, coord d). Its own kind 'rope'
        # (tiled like the fused-attention head grid, not like a flat 'ew').
        f32 = self.f32
        hs = D // nh
        half = hs // 2
        out_buf = self._buf((T, D), f32)
        # view (T,D) buffers as (nh, T, hs) strided (head-outermost transpose view).
        src3 = self._heads_view_of(src_buf, T, nh, hs)
        out3 = self._heads_view_of(out_buf, T, nh, hs)
        # (nh,T,hs) has strides [hs, D, 1]; split the last (hs) dim into two halves.
        lo = ir.StridedLayoutAttr.get(0, [hs, D, 1])  # first half, offset 0
        hi = ir.StridedLayoutAttr.get(half, [hs, D, 1])  # second half, offset half
        t_lo = ir.MemRefType.get((nh, T, half), f32, layout=lo)
        t_hi = ir.MemRefType.get((nh, T, half), f32, layout=hi)
        s1 = memref.subview(src3, [0, 0, 0], [nh, T, half], [1, 1, 1], result_type=t_lo)
        s2 = memref.subview(
            src3, [0, 0, half], [nh, T, half], [1, 1, 1], result_type=t_hi
        )
        o1 = memref.subview(out3, [0, 0, 0], [nh, T, half], [1, 1, 1], result_type=t_lo)
        o2 = memref.subview(
            out3, [0, 0, half], [nh, T, half], [1, 1, 1], result_type=t_hi
        )
        s1t = emit_buf_to_tensor(s1, restrict=True)
        s2t = emit_buf_to_tensor(s2, restrict=True)
        cos_t = emit_buf_to_tensor(cos, restrict=True)
        sin_t = emit_buf_to_tensor(sin, restrict=True)
        o1t = emit_buf_to_tensor(o1, restrict=True, writable=True)
        o2t = emit_buf_to_tensor(o2, restrict=True, writable=True)
        d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
        idn = affine_map(3, [d0, d1, d2])  # (head, t, coord)
        csm = affine_map(3, [d1, d2])  # cos/sin indexed by (t, coord)

        @linalg.generic(
            [s1t, s2t, cos_t, sin_t],
            [o1t, o2t],
            [idn, idn, csm, csm, idn, idn],
            [parallel, parallel, parallel],
        )
        def rot(a, b, co, si, _o1, _o2):
            r1 = arith.SubFOp(arith.MulFOp(a, co).result, arith.MulFOp(b, si).result)
            r2 = arith.AddFOp(arith.MulFOp(b, co).result, arith.MulFOp(a, si).result)
            return r1.result, r2.result

        bufferization.materialize_in_destination(
            None, rot[0], o1, restrict=True, writable=True
        )
        bufferization.materialize_in_destination(
            None, rot[1], o2, restrict=True, writable=True
        )
        self.kinds.append("rope")
        return emit_buf_to_tensor(out_buf, restrict=True)

    # ---- cast f32 (T,C) -> f16 (T,C), returning the MEMREF buffer (for views) ----
    def cast_f16_buf(self, x, T, C):
        par2 = self._par()
        buf = self._buf((T, C), self.f16)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([x], [out_t], [par2, par2], [parallel, parallel])
        def c(s, _o):
            return arith.TruncFOp(self.f16, s)

        bufferization.materialize_in_destination(
            None, c, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        return buf

    # ---- view a (T, H*hs) memref as (H, T, hs) -- no kernel, no data move ----
    def _heads_view_of(self, buf2d, T, H, hs):
        # Present the 2D (T, H*hs) projection buffer as a (H,T,hs) strided view:
        #   (T,H*hs) --memref.expand_shape--> (T,H,hs) [strides C,hs,1]
        #            --memref.transpose [1,0,2]--> (H,T,hs) [strides hs,C,1]
        # Both are pure layout ops (no compute, no kinds entry). When the fused
        # schedule tiles (1,wg_rows,0,0), the grid peels head h -> a 2D
        # memref<T x hs, strided<[C,1], offset:h*hs>> -> 2D load_nd (XeGPU supports
        # such strided block loads).
        C = H * hs
        et = buf2d.type.element_type
        exp_t = ir.MemRefType.get((T, H, hs), et)
        e = memref.expand_shape(
            exp_t, buf2d, [[0], [1, 2]], [], static_output_shape=[T, H, hs]
        )
        d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
        perm = ir.AffineMap.get(3, 0, [d1, d0, d2])  # (H,T,hs) <- (T,H,hs)
        layout = ir.StridedLayoutAttr.get(0, [hs, C, 1])
        res_t = ir.MemRefType.get((H, T, hs), et, layout=layout)
        return memref.transpose(res_t, e, perm)

    def heads_view(self, buf2d, T, H, hs):
        return emit_buf_to_tensor(self._heads_view_of(buf2d, T, H, hs), restrict=True)

    # ---- view (T, H*hs) as (n_kv, n_rep, T, hs) -- no kernel, no data move ----
    def _grouped_heads_view_of(self, buf2d, T, n_kv, n_rep, hs):
        # For GQA the Q head axis H = n_kv*n_rep is split so query head
        # h = kv*n_rep + rep. Presenting Q (and the attention output) as a 4D
        # (n_kv, n_rep, T, hs) view lets the QK^T/@V generics index the narrow
        # (n_kv, T, hs) K/V by the OUTER kv dim alone -- a projected permutation
        # (rep/query-row dropped), which vectorizes to a plain vector.contract.
        # A floordiv map (kv = h // n_rep over a flat H axis) would NOT vectorize.
        #   (T, H*hs) --expand--> (T, n_kv, n_rep, hs) --transpose [1,2,0,3]-->
        #   (n_kv, n_rep, T, hs).  Pure layout ops (no compute, no kinds entry).
        C = n_kv * n_rep * hs
        et = buf2d.type.element_type
        exp_t = ir.MemRefType.get((T, n_kv, n_rep, hs), et)
        e = memref.expand_shape(
            exp_t, buf2d, [[0], [1, 2, 3]], [], static_output_shape=[T, n_kv, n_rep, hs]
        )
        d0, d1, d2, d3 = (ir.AffineDimExpr.get(i) for i in range(4))
        perm = ir.AffineMap.get(
            4, 0, [d1, d2, d0, d3]
        )  # (n_kv,n_rep,T,hs)<-(T,n_kv,n_rep,hs)
        # (T,n_kv,n_rep,hs) row-major strides are [C, n_rep*hs, hs, 1]; permute them.
        layout = ir.StridedLayoutAttr.get(0, [n_rep * hs, hs, C, 1])
        res_t = ir.MemRefType.get((n_kv, n_rep, T, hs), et, layout=layout)
        return memref.transpose(res_t, e, perm)

    def grouped_heads_view(self, buf2d, T, n_kv, n_rep, hs):
        return emit_buf_to_tensor(
            self._grouped_heads_view_of(buf2d, T, n_kv, n_rep, hs), restrict=True
        )

    # ---- fused grouped-query attention core -----------------------------------
    # Q / output views are (n_kv, n_rep, T, hs); K / V views are the narrow
    # (n_kv, T, hs). GQA (grouped-query attention): K/V have n_kv <= H heads, and
    # query head h = kv*n_rep + rep reads KV head kv. Instead of materializing a
    # `repeat_kv` copy, QK^T and @V are linalg.generic ops that index the narrow
    # K/V by the OUTER kv dim ONLY -- a projected permutation (the n_rep and
    # query-row dims are simply dropped from the K/V map). That vectorizes to a
    # plain vector.contract, and after tiling (kv,rep) by 1 the per-head inner op
    # is byte-identical to plain MHA, so the fused-attention rewrite/annotations
    # are unchanged. (A floordiv map kv = h // n_rep over a flat H axis would NOT
    # vectorize -- it is not a projected permutation.)
    def attention_4d(self, Qh, Kh, Vh, n_kv, n_rep, T, hs, out_view, out_view_memref):
        # linalg op sequence: QK^T generic -> scale-mul -> softmax -> @V generic.
        # After the per-region fused tiling, these fuse into one scf.forall -> one
        # GPU kernel (the flash/online-softmax kernel). Counts as one 'fa'.
        f16 = self.f16
        scale = 1.0 / (hs**0.5)
        zero = arith.constant(f16, 0.0)

        # QK^T: qkt[kv,rep,i,j] = sum_k Q[kv,rep,i,k] * K[kv,j,k]  (no explicit K^T;
        # the generic contracts the head-dim k directly, K read as (kv, seq, head)).
        # iter dims: (kv d0, rep d1, i d2, j d3, k d4); k is the reduction.
        d0, d1, d2, d3, d4 = (ir.AffineDimExpr.get(d) for d in range(5))
        q_map = affine_map(5, [d0, d1, d2, d4])
        k_map = affine_map(5, [d0, d3, d4])  # drops rep/i -> projected permutation
        qkt_map = affine_map(5, [d0, d1, d2, d3])
        qkt_init = linalg.fill(zero, outs=[tensor.empty((n_kv, n_rep, T, T), f16)])

        @linalg.generic(
            [Qh, Kh],
            [qkt_init],
            [q_map, k_map, qkt_map],
            [parallel, parallel, parallel, parallel, reduction],
        )
        def qkt(qv, kv, _o):
            return arith.AddFOp(arith.MulFOp(qv, kv).result, _o)

        sc = arith.constant(f16, scale)
        scale_t = linalg.fill(sc, outs=[tensor.empty((n_kv, n_rep, T, T), f16)])
        scaled = linalg.mul(qkt, scale_t, outs=[tensor.empty((n_kv, n_rep, T, T), f16)])
        aw = linalg.softmax(
            result=[ir.RankedTensorType.get((n_kv, n_rep, T, T), f16)],
            input=scaled,
            output=tensor.empty((n_kv, n_rep, T, T), f16),
            dimension=3,  # softmax over the key-seq (last) dim
        )
        # @V: out[kv,rep,i,l] = sum_j aw[kv,rep,i,j] * V[kv,j,l]  (l = head_dim out,
        # j = key-seq reduction). Materialized f16 into the (n_kv,n_rep,T,hs) view.
        e0, e1, e2, e3, e4 = (ir.AffineDimExpr.get(d) for d in range(5))
        aw_map = affine_map(5, [e0, e1, e2, e4])
        v_map = affine_map(5, [e0, e4, e3])  # drops rep/i -> projected permutation
        out_map = affine_map(5, [e0, e1, e2, e3])
        out_filled = linalg.fill(zero, outs=[out_view])

        @linalg.generic(
            [aw, Vh],
            [out_filled],
            [aw_map, v_map, out_map],
            [parallel, parallel, parallel, parallel, reduction],
        )
        def out(av, vv, _o):
            return arith.AddFOp(arith.MulFOp(av, vv).result, _o)

        bufferization.materialize_in_destination(
            None, out, out_view_memref, restrict=True, writable=True
        )
        self.kinds.append("fa")

    # ---- fused grouped-query attention(rms_f32 (T,C) f32) -> (T,C) f16 ----
    # Emits non-causal linalg; the causal mask (if enabled) is injected later by
    # the fused-attention transform op in the schedule, not here.
    def fused_attention(self, x, wq, wk, wv, cos, sin, T, C, H, n_kv):
        # True grouped-query attention via the flash kernel, no on-device
        # head-transpose or repeat_kv kernel. Flow:
        #   x(f32) -cast-> f16 -q proj-> (T,C) f32 -RoPE-> -cast-> f16
        #                    -grouped_heads_view (free)-> (n_kv,n_rep,T,hs) Q view
        #                  -k/v proj-> (T,kv_dim) f32; k gets RoPE too -cast-> f16
        #                    -heads_view-> (n_kv,T,hs) K/V
        #   -> attention_4d (fused flash kernel, K/V shared across the n_rep group)
        #   -> @V written into a (T,C) f16 buffer via its (n_kv,n_rep,T,hs) view ->
        #   return (T,C) f16.
        # RoPE (rotary pos emb) is applied to q and k (NOT v) on the f32 projection
        # output, before the f16 cast, per head. cos/sin are (T, hs/2) f32 args.
        hs = C // H
        n_rep = H // n_kv  # query heads sharing each KV head
        kv_dim = n_kv * hs  # narrow K/V feature width (GQA: kv_dim <= C)
        x16 = self.cast_f16(x, T, C)  # ew
        qp = self._buf((T, C), self.f32)
        self.matmul(x16, wq, T, C, out_buf=qp)  # mm -> f32 q projection
        qbuf = self.cast_f16_buf(self.rope(qp, cos, sin, T, C, H), T, C)  # ew(rope), ew
        kp = self._buf((T, kv_dim), self.f32)
        self.matmul(x16, wk, T, kv_dim, out_buf=kp)  # mm -> f32 k projection
        kbuf = self.cast_f16_buf(
            self.rope(kp, cos, sin, T, kv_dim, n_kv), T, kv_dim
        )  # ew(rope), ew
        vbuf = self.cast_f16_buf(self.matmul(x16, wv, T, kv_dim), T, kv_dim)  # mm, ew
        Qh = self.grouped_heads_view(qbuf, T, n_kv, n_rep, hs)  # (n_kv,n_rep,T,hs)
        Kh = self.heads_view(kbuf, T, n_kv, hs)  # (n_kv,T,hs) strided view
        Vh = self.heads_view(vbuf, T, n_kv, hs)
        out_buf = self._buf((T, C), self.f16)
        out_view_memref = self._grouped_heads_view_of(out_buf, T, n_kv, n_rep, hs)
        out_view = emit_buf_to_tensor(out_view_memref, restrict=True, writable=True)
        self.attention_4d(
            Qh, Kh, Vh, n_kv, n_rep, T, hs, out_view, out_view_memref
        )  # fa
        return emit_buf_to_tensor(out_buf, restrict=True)  # (T,C) f16


# ---------------------------------------------------------------------------
# Payload assembly -- wire the Builder ops into a complete MLIR function.
# `build_llama_payload` creates one `func.func` (the "payload") whose arguments
# are the input + all weights (as device memrefs) and whose body is the op graph.
# Returns (module, kinds) where `kinds` drives the schedule.
# ---------------------------------------------------------------------------


def _emit_block_llama(bld, x, w, cos, sin, T, C, hidden, H, n_kv, eps, out_buf=None):
    """Emit one Llama transformer block (fused GQA + RoPE, no bias; causal
    masking, if any, is applied by the schedule's fused-attention transform).
    `w` weight keys: an (attention RMSNorm weight), wq,wk,wv, wo,
    fn (ffn RMSNorm weight), w1,w2,w3.  wk/wv are narrow (C, n_kv*hs) for GQA.
    cos/sin are the shared (T, hs/2) RoPE tables.
        h   = x + wo( GQA( RoPE(rms_attn(x)) ) )  # attention sublayer + residual
        out = h + swiglu( rms_ffn(h) )            # FFN sublayer + residual
    SwiGLU: w2( silu(z@w1) * (z@w3) ).
    """
    # ---- attention sublayer: h = x + wo(GQA(RoPE(rms(x)))) ----
    rms1 = bld.rmsnorm(x, w["an"], T, C, eps)
    attn16 = bld.fused_attention(
        rms1, w["wq"], w["wk"], w["wv"], cos, sin, T, C, H, n_kv
    )  # f16 (T,C)
    proj = bld.matmul(attn16, w["wo"], T, C)  # (T,C) f32, no bias
    h = bld.add(x, proj, T, C)
    # ---- FFN sublayer: out = h + swiglu(rms(h)) ----
    rms2 = bld.rmsnorm(h, w["fn"], T, C, eps)
    z16 = bld.cast_f16(rms2, T, C)
    gate = bld.matmul(z16, w["w1"], T, hidden)  # z@w1 -> (T,hidden) f32
    gate = bld.silu(gate, T, hidden)  # silu(z@w1)
    up = bld.matmul(z16, w["w3"], T, hidden)  # z@w3 -> (T,hidden) f32
    prod = bld.mul(gate, up, T, hidden)  # silu(z@w1) * (z@w3)
    prod16 = bld.cast_f16(prod, T, hidden)
    o = bld.matmul(prod16, w["w2"], T, C)  # (T,C) f32
    return bld.add(h, o, T, C, out_buf=out_buf)


def build_llama_payload(func_name, T, C, hidden, vocab, n_layers, H, n_kv, eps=1e-5):
    """Full Llama-3 forward as one module, fused grouped-query attention per block
    (n_kv KV heads, RoPE on q/k, causal masking applied later by the schedule),
    RMSNorm, SwiGLU FFN, no biases. Embeddings done host-side. Returns
    (module, kinds, mm_shapes) -- mm_shapes is the (M,N,K) of each matmul in build
    order, so the schedule can pick per-matmul DPAS params (K/V projections are
    narrow, N = n_kv*hs < C, and need different tiling than the wide matmuls)."""
    f32, f16 = F32(), F16()
    kv_dim = n_kv * (C // H)  # narrow K/V feature width (GQA: kv_dim <= C)
    half = (C // H) // 2  # RoPE cos/sin table width (head_dim/2)
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)  # input activations f32
    cs_t = ir.MemRefType.get((T, half), f32)  # RoPE cos/sin tables (T, hs/2) f32
    n_t = ir.MemRefType.get((C,), f32)  # RMSNorm weight vectors (C,) f32
    wq_t = ir.MemRefType.get((C, C), f16)  # q projection weight f16
    wkv_t = ir.MemRefType.get((C, kv_dim), f16)  # narrow k/v projection weights f16
    wo_t = ir.MemRefType.get((C, C), f16)  # attention output proj weight f16
    w1_t = ir.MemRefType.get((C, hidden), f16)  # SwiGLU gate/up projections f16
    w2_t = ir.MemRefType.get((hidden, C), f16)  # SwiGLU down projection f16
    lmw_t = ir.MemRefType.get((C, vocab), f16)  # output (lm_head) weight f16
    out_t = ir.MemRefType.get((T, vocab), f32)  # output logits f32
    # per-layer arg types: an, wq,wk,wv, wo, fn, w1,w2,w3 (9) -- no bias, no mask.
    per_layer = [n_t, wq_t, wkv_t, wkv_t, wo_t, n_t, w1_t, w2_t, w1_t]
    fargs = [out_t, x_t, cs_t, cs_t]  # output, input, RoPE cos, RoPE sin
    for _ in range(n_layers):
        fargs += per_layer
    fargs += [n_t, lmw_t]  # final RMSNorm weight, output weight
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):

        @func_cif(*fargs, name=func_name)
        def payload(*args):
            output = args[0]
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(args[1], restrict=True)
            cos, sin = args[2], args[3]  # raw memrefs (rope views them itself)
            idx = 4
            layer_w = []
            keys = ["an", "wq", "wk", "wv", "wo", "fn", "w1", "w2", "w3"]
            for _ in range(n_layers):
                w = {
                    k: emit_buf_to_tensor(args[idx + i], restrict=True)
                    for i, k in enumerate(keys)
                }
                idx += len(keys)
                layer_w.append(w)
            fn_w = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1
            lmw = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1

            h = x
            for w in layer_w:
                h = _emit_block_llama(bld, h, w, cos, sin, T, C, hidden, H, n_kv, eps)
            hf = bld.rmsnorm(h, fn_w, T, C, eps)
            hf16 = bld.cast_f16(hf, T, C)
            bld.matmul(hf16, lmw, T, vocab, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)

        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds, bld.mm_shapes

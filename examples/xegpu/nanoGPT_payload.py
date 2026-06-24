"""Generate the nano-GPT / GPT-2-style forward pass payload at the linalg level.

This is STAGE 1 -- the PAYLOAD ("WHAT to compute"): an MLIR module describing
the GPT payload at linalg level.

  -> class `Builder` (emits one op at a time) and `build_gpt_fused_payload`
     (assembles ops into ffn / attn / block / full-gpt).

C = n_embd, the embedding/channel width (C=256 in this example).

Architecture: each transformer block is
    a = x + attn_proj( MultiHeadAttention( ln1(x) ) )       # attention sublayer
    y = a + ffn( ln2(a) )                                    # MLP (feed-forward) sublayer
    ffn(z) = Linear(C, 4C) -> ReLU -> Linear(4C, C)         # the MLP: two matmuls
and the full model is
    x = token_emb + pos_emb            # embeddings (done host-side)
    for _ in range(n_layer): x = Block(x)
    x = ln_f(x); logits = x @ lm_head
TRUE multi-head: H heads of head_size = C/H = 64, computed by ONE fused
flash-attention kernel per block (NON-CAUSAL for now).
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


def F32():  # 32-bit float (used for accumulation / norms)
    return ir.F32Type.get()


def F16():  # 16-bit float (required by the GPU matmul units)
    return ir.F16Type.get()


# =============================================================================
# PAYLOAD: describe WHAT to compute (high-level linalg ops; no tiling/XeGPU yet)
# =============================================================================
# Each Builder method emits ONE high-level op that writes its result into a fresh
# on-device buffer (`gpu.alloc`), and returns a tensor "view" of that buffer for
# the next op to read. Because each op writes a distinct device buffer, each will
# become its OWN GPU kernel later; the buffers are the on-device handoff between
# kernels (kernel N writes buffer B, kernel N+1 reads B -- no host round-trip).
#
# dtype convention: the GPU matmul (DPAS) hardware needs f16 inputs and produces
# an f32 result. LayerNorm/softmax run in f32. So between a norm/softmax and a
# matmul we insert an explicit f32->f16 `cast` op (its own kernel).
# =============================================================================
class Builder:
    """Emits the model's ops and remembers the order/kind of each one.

    `kinds` is the crucial bookkeeping: an ordered list, one entry per op emitted,
    recording its "class" so the SCHEDULE (stage 2) can later tile and annotate
    each kernel correctly. Classes:
      'mm'  = matmul (linalg.matmul)          -> DPAS systolic-array kernel
      'ln'  = layernorm (3 generics + 2 fills) -> reduction kernel (uses shared mem)
      'fa'  = flash multi-head attention -> ONE kernel (QK^T->softmax->@V,
              online-softmax over K/V tiles); see attention_4d + the fused-attention
              schedule helpers. (Softmax lives INSIDE this kernel, not as its own.)
      'ew'  = elementwise (cast / bias / relu / residual) -> simple row-parallel kernel
    The op build order in the payload == the order of `kinds` == the order the
    kernels appear in the final module, which is how the schedule matches them up.
    """

    def __init__(self, T):
        self.T = T
        self.f32, self.f16 = F32(), F16()
        self.kinds = []  # ordered kernel classes (see docstring)
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
        if out_buf is not None:  # caller gave the final output buffer
            return None
        return emit_buf_to_tensor(buf, restrict=True)

    # ---- layernorm(x (M,N) f32, gamma,beta (N,)) -> (M,N) f32 buffer ----
    def layernorm(self, x, gamma, beta, M, N, eps=1e-5):
        # LayerNorm normalizes each ROW (length N) to mean 0 / variance 1, then
        # scales by gamma and shifts by beta. Built from 3 linalg.generic ops:
        #   (1) mean_sum[i] = sum_j x[i,j]                 (row reduction)
        #   (2) var_sum[i]  = sum_j (x[i,j]-mean_i)^2      (row reduction)
        #   (3) out[i,j]    = (x[i,j]-mean_i)*rsqrt(var_i+eps)*gamma[j] + beta[j]
        # Affine maps describe each operand's access pattern:
        #   par2  (d0,d1)->(d0,d1) : full 2-D elementwise
        #   red2  (d0,d1)->(d0)    : reduce over j -> one value per row
        #   bias2 (d0,d1)->(d1)    : gamma/beta indexed by column only
        f32 = self.f32
        par2, red2 = self._par(), affine_map(2, [ir.AffineDimExpr.get(0)])
        bias2 = affine_map(2, [ir.AffineDimExpr.get(1)])
        inv_n = arith.constant(f32, 1.0 / float(N))
        eps_c = arith.constant(f32, eps)
        zero = arith.constant(f32, 0.0)
        # (1) row sums -> mean_sum (linalg.fill zeroes the accumulator first)
        mean_acc = linalg.fill(zero, outs=[tensor.empty((M,), f32)])

        @linalg.generic([x], [mean_acc], [par2, red2], [parallel, reduction])
        def mean_sum(v, acc):
            return arith.AddFOp(v, acc)

        # (2) sum of squared deviations -> var_sum (mean_i = mean_sum_i / N)
        var_acc = linalg.fill(zero, outs=[tensor.empty((M,), f32)])

        @linalg.generic(
            [x, mean_sum], [var_acc], [par2, red2, red2], [parallel, reduction]
        )
        def var_sum(v, m_sum, acc):
            mean = arith.MulFOp(m_sum, inv_n).result
            c = arith.SubFOp(v, mean).result
            return arith.AddFOp(arith.MulFOp(c, c).result, acc)

        # (3) normalize + scale + shift -> output
        buf = self._buf((M, N), f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic(
            [x, mean_sum, var_sum, gamma, beta],
            [out_t],
            [par2, red2, red2, bias2, bias2, par2],
            [parallel, parallel],
        )
        def normed(v, m_sum, v_sum, g, b, _o):
            mean = arith.MulFOp(m_sum, inv_n).result
            var = arith.MulFOp(v_sum, inv_n).result
            inv_std = math.rsqrt(arith.AddFOp(var, eps_c).result)
            c = arith.SubFOp(v, mean).result
            return arith.AddFOp(
                arith.MulFOp(arith.MulFOp(c, inv_std).result, g).result, b
            )

        bufferization.materialize_in_destination(
            None, normed, buf, restrict=True, writable=True
        )
        self.kinds.append("ln")
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

    # ---- bias add (+ optional relu): out = max(x + bias, 0)?  x (M,N) f32, bias (N,) ----
    def bias(self, x, bias_vec, M, N, relu=False, out_buf=None):
        par2 = self._par()
        bias2 = affine_map(2, [ir.AffineDimExpr.get(1)])
        zero = arith.constant(self.f32, 0.0)
        buf = out_buf if out_buf is not None else self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic(
            [x, bias_vec], [out_t], [par2, bias2, par2], [parallel, parallel]
        )
        def b(v, bb, _o):
            s = arith.AddFOp(v, bb).result
            if relu:
                return arith.MaximumFOp(s, zero)
            return arith.AddFOp(s, zero)  # identity wrap so the op has a body

        bufferization.materialize_in_destination(
            None, b, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        if out_buf is not None:
            return None
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

    # ---- view a (T, H*hs) MEMREF as (H, T, hs) -- NO kernel, NO data move ----
    def _heads_view_of(self, buf2d, T, H, hs):
        #  We present the 2D
        # (T, H*hs) projection buffer as a (H,T,hs) STRIDED memref VIEW:
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

    # ---- fused multi-head attention core on 3D (H,T,hs) f16 -> (H,T,hs) f16 ----
    # (named attention_4d because it is the canonical 4D (Z,H,T,hs) attention
    #  algorithm with the batch dim Z=1 FOLDED OUT: one sequence, so (1,H,T,hs)
    #  collapses to (H,T,hs) and linalg.batch_matmul treats H as the batch axis.)
    def attention_4d(self, Qh, Kh, Vh, H, T, hs, out_view, out_view_memref):
        # Emits the SAME linalg op sequence as generate_gpu_attention_payload
        # (batch_matmul QK^T -> scale-mul -> softmax -> batch_matmul @V), so the
        # fused-attention schedule's matchers/rewrite apply verbatim. After the
        # per-region fused tiling, ALL these ops fuse into ONE scf.forall -> ONE
        # GPU kernel (the flash/online-softmax kernel). Counts as one 'fa'.
        # Inputs Qh/Kh/Vh are (H,T,hs) f16 strided VIEWS (heads_view); the @V result
        # is materialized into `out_view`, a (H,T,hs) strided view of a (T,C) buffer,
        # so the merge back to 2D is also a free view (no from_heads kernel).
        f16 = self.f16
        scale = 1.0 / (hs**0.5)
        zero = arith.constant(f16, 0.0)
        # K^T: (H,T,hs) -> (H,hs,T). Lowers to a 2D vector.transpose per head (the
        # grid peels H), exactly like the standalone -- f16 is fine here.
        Kt = linalg.transpose(
            Kh, outs=[tensor.empty((H, hs, T), f16)], permutation=[0, 2, 1]
        )
        qkt_init = linalg.fill(zero, outs=[tensor.empty((H, T, T), f16)])
        qkt = linalg.batch_matmul(Qh, Kt, outs=[qkt_init])
        sc = arith.constant(f16, scale)
        scale_t = linalg.fill(sc, outs=[tensor.empty((H, T, T), f16)])
        scaled = linalg.mul(qkt, scale_t, outs=[tensor.empty((H, T, T), f16)])
        aw = linalg.softmax(
            result=[ir.RankedTensorType.get((H, T, T), f16)],
            input=scaled,
            output=tensor.empty((H, T, T), f16),
            dimension=2,
        )
        # @V: (H,T,T) @ (H,T,hs) -> (H,T,hs) f16, materialized into the (T,C) view.
        out_filled = linalg.fill(zero, outs=[out_view])
        out = linalg.batch_matmul(aw, Vh, outs=[out_filled])
        bufferization.materialize_in_destination(
            None, out, out_view_memref, restrict=True, writable=True
        )
        self.kinds.append("fa")

    # ---- fused multi-head attention(ln_f32 (T,C) f32) -> (T,C) f16, NON-CAUSAL ----
    def fused_attention(self, x, wq, wk, wv, T, C, H):
        # True multi-head attention via the flash kernel, with NO on-device
        # head-transpose kernel. Flow:
        #   x(f32) -cast-> f16 -q/k/v proj-> (T,C) f16 buffers -heads_view (free)->
        #   (H,T,hs) strided views -> attention_4d (fused flash kernel) -> @V written
        #   into a (T,C) f16 buffer via its (H,T,hs) view -> return that (T,C) f16.
        hs = C // H
        x16 = self.cast_f16(x, T, C)  # ew
        qbuf = self.cast_f16_buf(
            self.matmul(x16, wq, T, C), T, C
        )  # mm, ew -> (T,C) f16 memref
        kbuf = self.cast_f16_buf(self.matmul(x16, wk, T, C), T, C)  # mm, ew
        vbuf = self.cast_f16_buf(self.matmul(x16, wv, T, C), T, C)  # mm, ew
        Qh = self.heads_view(qbuf, T, H, hs)  # (H,T,hs) strided view (free)
        Kh = self.heads_view(kbuf, T, H, hs)
        Vh = self.heads_view(vbuf, T, H, hs)
        # Output (T,C) f16 buffer, viewed as (H,T,hs) for the @V store.
        out_buf = self._buf((T, C), self.f16)
        out_view_memref = self._heads_view_of(out_buf, T, H, hs)
        out_view = emit_buf_to_tensor(out_view_memref, restrict=True, writable=True)
        self.attention_4d(
            Qh, Kh, Vh, H, T, hs, out_view, out_view_memref
        )  # fa, writes out_buf
        return emit_buf_to_tensor(out_buf, restrict=True)  # (T,C) f16


# ---------------------------------------------------------------------------
# PAYLOAD ASSEMBLY -- wire the Builder ops into a complete MLIR function.
# `build_gpt_fused_payload` creates one `func.func` (the "payload") whose arguments
# are the input + all weights (as device memrefs) and whose body is the op graph.
# `emit_buf_to_tensor` turns a memref argument into a tensor the ops can read;
# `func_cif` makes the function callable from C/the Runner. Returns (module,
# kinds) where `kinds` drives the schedule.
# ---------------------------------------------------------------------------


def _emit_block_fused(bld, x, w, T, C, hidden, H, eps, out_buf=None):
    """Emit ONE transformer block whose attention sublayer is the FUSED multi-head
    flash kernel (NON-CAUSAL, no mask). `w` weight keys: g1,b1n, wq,wk,wv, wp,bp,
    g2,b2n, w1,bb1,w2,bb2. wq/wk/wv/wp are full (C,C)."""
    # ---- attention sublayer: a = x + proj(fused_attn(ln1(x))) ----
    ln1 = bld.layernorm(x, w["g1"], w["b1n"], T, C, eps)
    attn16 = bld.fused_attention(ln1, w["wq"], w["wk"], w["wv"], T, C, H)  # f16 (T,C)
    proj = bld.matmul(attn16, w["wp"], T, C)
    proj = bld.bias(proj, w["bp"], T, C, relu=False)
    a = bld.add(x, proj, T, C)
    # ---- FFN sublayer: y = a + ffn(ln2(a)) ----
    ln2 = bld.layernorm(a, w["g2"], w["b2n"], T, C, eps)
    ln2_16 = bld.cast_f16(ln2, T, C)
    h = bld.matmul(ln2_16, w["w1"], T, hidden)
    h = bld.bias(h, w["bb1"], T, hidden, relu=True)
    h16 = bld.cast_f16(h, T, hidden)
    o = bld.matmul(h16, w["w2"], T, C)
    o = bld.bias(o, w["bb2"], T, C, relu=False)
    return bld.add(a, o, T, C, out_buf=out_buf)


def build_gpt_fused_payload(func_name, T, C, hidden, vocab, n_layer, H, eps=1e-5):
    """Full nanoGPT forward as ONE module, with FUSED multi-head attention per block.
    Multi-head (H heads, flash attention), NON-CAUSAL (no mask), wq/wk/wv/wp are
    (C,C). Embeddings done host-side. Returns (module, kinds)."""
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((T, C), f32)  # input activations (256,256) f32
    g_t = ir.MemRefType.get((C,), f32)  # layernorm gamma/beta vectors (256,) f32
    wqkv_t = ir.MemRefType.get((C, C), f16)  # q/k/v projection weights (256,256) f16
    wproj_t = ir.MemRefType.get(
        (C, C), f16
    )  # attention output proj weight (256,256) f16
    bvec_t = ir.MemRefType.get((C,), f32)  # bias vectors (256,) f32
    w1_t = ir.MemRefType.get((C, hidden), f16)  # FFN up-projection (256,1024) f16
    b1_t = ir.MemRefType.get((hidden,), f32)  # FFN hidden bias (1024,) f32
    w2_t = ir.MemRefType.get((hidden, C), f16)  # FFN down-projection (1024,256) f16
    lmw_t = ir.MemRefType.get((C, vocab), f16)  # lm_head weight (256,256) f16
    lmb_t = ir.MemRefType.get((vocab,), f32)  # lm_head bias (256,) f32
    out_t = ir.MemRefType.get((T, vocab), f32)  # output logits (256,256) f32
    # per-layer arg types: g1,b1n, wq,wk,wv, wp,bp, g2,b2n, w1,bb1,w2,bb2 (13) -- NO mask.
    per_layer = [
        g_t,
        g_t,
        wqkv_t,
        wqkv_t,
        wqkv_t,
        wproj_t,
        bvec_t,
        g_t,
        g_t,
        w1_t,
        b1_t,
        w2_t,
        bvec_t,
    ]
    from lighthouse.utils.mlir import func_cif

    fargs = [out_t, x_t]
    for _ in range(n_layer):
        fargs += per_layer
    fargs += [g_t, g_t, lmw_t, lmb_t]  # ln_f gamma/beta, lm_head W,b (no mask)
    bld = Builder(T)
    with ir.InsertionPoint(mod.body):

        @func_cif(*fargs, name=func_name)
        def payload(*args):
            output = args[0]
            emit_buf_to_tensor(output, restrict=True, writable=True)
            x = emit_buf_to_tensor(args[1], restrict=True)
            idx = 2
            layer_w = []
            keys = [
                "g1",
                "b1n",
                "wq",
                "wk",
                "wv",
                "wp",
                "bp",
                "g2",
                "b2n",
                "w1",
                "bb1",
                "w2",
                "bb2",
            ]
            for _ in range(n_layer):
                w = {
                    k: emit_buf_to_tensor(args[idx + i], restrict=True)
                    for i, k in enumerate(keys)
                }
                idx += len(keys)
                layer_w.append(w)
            gf_g = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1
            gf_b = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1
            lmw = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1
            lmb = emit_buf_to_tensor(args[idx], restrict=True)
            idx += 1

            h = x
            for w in layer_w:
                h = _emit_block_fused(bld, h, w, T, C, hidden, H, eps)
            hf = bld.layernorm(h, gf_g, gf_b, T, C, eps)
            hf16 = bld.cast_f16(hf, T, C)
            logits = bld.matmul(hf16, lmw, T, vocab)
            bld.bias(logits, lmb, T, vocab, relu=False, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)

        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds

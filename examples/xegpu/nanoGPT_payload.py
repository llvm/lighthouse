"""Generate the nano-GPT / GPT-2-style forward pass payload at the linalg level.

This is stage 1 -- the payload ("what to compute"): an MLIR module describing the
GPT forward pass with linalg-level ops that write into device (gpu.alloc) buffers.

  -> class `Builder` (emits one op at a time) and `build_gpt_fused_payload`
     (assembles ops into ffn / attn / block / full-gpt).

n_embd = the embedding/channel width (n_embd=256 in this example).

Architecture: each transformer block is
    a = x + attn_proj( MultiHeadAttention( ln1(x) ) )       # attention sublayer
    y = a + ffn( ln2(a) )                                    # MLP (feed-forward) sublayer
    ffn(z) = Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd, n_embd)  # the MLP: two matmuls
and the full model is
    x = token_emb + pos_emb            # embeddings (done host-side)
    for _ in range(n_layer): x = Block(x)
    x = ln_f(x); logits = x @ lm_head
Multi-head attention uses n_head heads of d_head = n_embd/n_head = 64, computed by one fused
flash-attention kernel per block (non-causal for now).
"""

from mlir import ir
from mlir.dialects import linalg, bufferization, tensor, arith, gpu, math, memref

from lighthouse.ingress.mlir_gen.utils import (
    emit_buf_to_tensor,
    affine_map,
    parallel,
    reduction,
)
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.gpu_layer_norm_payload import emit_layer_norm_generics
from lighthouse.ingress.mlir_gen.named import times_weights
from lighthouse.utils.mlir import func_cif


def F32():  # 32-bit float (used for accumulation / norms)
    return ir.F32Type.get()


def F16():  # 16-bit float (required by the GPU matmul units)
    return ir.F16Type.get()


# =============================================================================
# PAYLOAD: describe WHAT to compute (high-level linalg ops; no tiling/XeGPU yet)
# =============================================================================
# Each Builder method emits one high-level op that writes its result into a fresh
# on-device buffer (`gpu.alloc`), and returns a tensor "view" of that buffer for
# the next op to read. Because each op writes a distinct device buffer, each will
# become its own GPU kernel later; the buffers are the on-device handoff between
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
      'fa'  = flash multi-head attention -> one kernel (QK^T->softmax->@V,
              online-softmax over K/V tiles); see attention_4d + the fused-attention
              schedule helpers. (Softmax lives INSIDE this kernel, not as its own.)
      'ew'  = elementwise (cast / bias / relu / residual) -> simple row-parallel kernel
    The op build order in the payload == the order of `kinds` == the order the
    kernels appear in the final module, which is how the schedule matches them up.
    """

    def __init__(self):
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
        # Row-wise LayerNorm: normalize each row to mean 0 / var 1, scale by gamma,
        # shift by beta. The 3-generic core (mean/var reductions + normalize) is
        # shared with the standalone gpu_layer_norm payload; here we just wrap it
        # to write into a device (gpu.alloc) buffer and record the kernel kind.
        buf = self._buf((M, N), self.f32)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)
        normed = emit_layer_norm_generics(x, gamma, beta, out_t, N, self.f32, eps)
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

    # ---- cast f32 (n_ctx,n_embd) -> f16 (n_ctx,n_embd), returning the MEMREF buffer (for views) ----
    def cast_f16_buf(self, x, n_ctx, n_embd):
        par2 = self._par()
        buf = self._buf((n_ctx, n_embd), self.f16)
        out_t = emit_buf_to_tensor(buf, restrict=True, writable=True)

        @linalg.generic([x], [out_t], [par2, par2], [parallel, parallel])
        def c(s, _o):
            return arith.TruncFOp(self.f16, s)

        bufferization.materialize_in_destination(
            None, c, buf, restrict=True, writable=True
        )
        self.kinds.append("ew")
        return buf

    # ---- view a (n_ctx, n_head*d_head) memref as (n_head, n_ctx, d_head) -- no kernel, no data move ----
    def _heads_view_of(self, buf2d, n_ctx, n_head, d_head):
        #  We present the 2D
        # (n_ctx, n_head*d_head) projection buffer as a (n_head,n_ctx,d_head) STRIDED memref VIEW:
        #   (n_ctx,n_head*d_head) --memref.expand_shape--> (n_ctx,n_head,d_head) [strides n_embd,d_head,1]
        #            --memref.transpose [1,0,2]--> (n_head,n_ctx,d_head) [strides d_head,n_embd,1]
        # Both are pure layout ops (no compute, no kinds entry). When the fused
        # schedule tiles (1,wg_rows,0,0), the grid peels head h -> a 2D
        # memref<n_ctx x d_head, strided<[n_embd,1], offset:h*d_head>> -> 2D load_nd (XeGPU supports
        # such strided block loads).
        n_embd = n_head * d_head
        et = buf2d.type.element_type
        exp_t = ir.MemRefType.get((n_ctx, n_head, d_head), et)
        e = memref.expand_shape(
            exp_t, buf2d, [[0], [1, 2]], [], static_output_shape=[n_ctx, n_head, d_head]
        )
        d0, d1, d2 = (ir.AffineDimExpr.get(i) for i in range(3))
        perm = ir.AffineMap.get(
            3, 0, [d1, d0, d2]
        )  # (n_head,n_ctx,d_head) <- (n_ctx,n_head,d_head)
        layout = ir.StridedLayoutAttr.get(0, [d_head, n_embd, 1])
        res_t = ir.MemRefType.get((n_head, n_ctx, d_head), et, layout=layout)
        return memref.transpose(res_t, e, perm)

    def heads_view(self, buf2d, n_ctx, n_head, d_head):
        return emit_buf_to_tensor(
            self._heads_view_of(buf2d, n_ctx, n_head, d_head), restrict=True
        )

    # ---- fused multi-head attention core on 3D (n_head,n_ctx,d_head) f16 -> (n_head,n_ctx,d_head) f16 ----
    # (named attention_4d because it is the canonical 4D (batch_size,n_head,n_ctx,d_head) attention
    #  algorithm with the batch dim batch_size=1 FOLDED OUT: one sequence, so (1,n_head,n_ctx,d_head)
    #  collapses to (n_head,n_ctx,d_head) and linalg.batch_matmul treats n_head as the batch axis.)
    def attention_4d(
        self, Qh, Kh, Vh, n_head, n_ctx, d_head, out_view, out_view_memref
    ):
        # batch_matmul QK^T -> scale-mul -> softmax -> batch_matmul @V, with the
        # softmax spelled out in the flash-attention form:
        #
        #   m   = max_j s          l   = sum_j p
        #   p   = exp(s - m)       o   = p @ V         out = o / l
        #
        # i.e. the normalizing divide comes *after* the contraction. That is
        # algebraically identical to `softmax(s) @ V` -- dividing by the per-row `l`
        # commutes with a contraction that reduces the other axis -- but it leaves
        # the `max -> exp -> {sum, @V}` dependency chain explicit, which is what
        # `transform_ext.fuse_dependant_reduction_ops` consumes to derive the online
        # one-pass loop (see `_fuse_attention_in_region` in nanoGPT_schedule.py).
        # A `linalg.softmax` would not do: its decomposition normalizes *before* the
        # contraction, leaving @V reading the normalized P and breaking the chain.
        #
        # Same chain shape as `generate_gpu_attention_payload`, but not the same ops:
        # this one stays f16 throughout and keeps @V a named `linalg.batch_matmul`,
        # while that one accumulates in f32 and writes @V as a `linalg.generic` so the
        # narrowing of P can sit in its body. Hence the extra `generalize` on the
        # `_fuse_attention_in_region` path -- the fusion needs @V as a generic.
        #
        # After the per-region fused tiling, all these ops fuse into one scf.forall
        # -> one GPU kernel (the flash/online-softmax kernel). Counts as one 'fa'.
        # Inputs Qh/Kh/Vh are (n_head,n_ctx,d_head) f16 strided views (heads_view); the @V result
        # is materialized into `out_view`, a (n_head,n_ctx,d_head) strided view of a (n_ctx,n_embd) buffer,
        # so the merge back to 2D is also a free view (no from_heads kernel).
        f16 = self.f16
        scale = 1.0 / (d_head**0.5)
        zero = arith.constant(f16, 0.0)
        # K^T: (n_head,n_ctx,d_head) -> (n_head,d_head,n_ctx). Lowers to a 2D vector.transpose per head (the
        # grid peels n_head), exactly like the standalone -- f16 is fine here.
        Kt = linalg.transpose(
            Kh, outs=[tensor.empty((n_head, d_head, n_ctx), f16)], permutation=[0, 2, 1]
        )
        qkt_init = linalg.fill(zero, outs=[tensor.empty((n_head, n_ctx, n_ctx), f16)])
        qkt = linalg.batch_matmul(Qh, Kt, outs=[qkt_init])
        sc = arith.constant(f16, scale)
        scale_t = linalg.fill(sc, outs=[tensor.empty((n_head, n_ctx, n_ctx), f16)])
        scaled = linalg.mul(
            qkt, scale_t, outs=[tensor.empty((n_head, n_ctx, n_ctx), f16)]
        )

        # (head, row, col) -> (head, row, col) and -> (head, row): the per-row
        # statistics are broadcast over the reduced axis.
        dims = [ir.AffineDimExpr.get(i) for i in range(3)]
        ew_map = affine_map(3, dims)
        row_map = affine_map(3, dims[:2])
        row_shape = (n_head, n_ctx)

        # m = max_j s
        neg_inf = arith.constant(f16, float("-inf"))
        max_acc = linalg.fill(neg_inf, outs=[tensor.empty(row_shape, f16)])

        @linalg.generic(
            [scaled], [max_acc], [ew_map, row_map], [parallel, parallel, reduction]
        )
        def row_max(s, acc):
            return arith.MaximumFOp(s, acc)

        # p = exp(s - m), read by both the row sum and the @V contraction.
        @linalg.generic(
            [scaled, row_max],
            [tensor.empty((n_head, n_ctx, n_ctx), f16)],
            [ew_map, row_map, ew_map],
            [parallel, parallel, parallel],
        )
        def probs(s, m, out):
            return math.ExpOp(arith.SubFOp(s, m).result)

        # l = sum_j p
        sum_acc = linalg.fill(zero, outs=[tensor.empty(row_shape, f16)])

        @linalg.generic(
            [probs], [sum_acc], [ew_map, row_map], [parallel, parallel, reduction]
        )
        def row_sum(p, acc):
            return arith.AddFOp(p, acc)

        # @V: (n_head,n_ctx,n_ctx) @ (n_head,n_ctx,d_head) -> (n_head,n_ctx,d_head)
        # f16, still unnormalized.
        unnorm_init = linalg.fill(
            zero, outs=[tensor.empty((n_head, n_ctx, d_head), f16)]
        )
        unnormalized = linalg.batch_matmul(probs, Vh, outs=[unnorm_init])

        # out = o / l, the deferred normalization, materialized into the
        # (n_ctx,n_embd) view. `l` broadcasts over d_head, the contraction's free
        # axis.
        @linalg.generic(
            [unnormalized, row_sum],
            [out_view],
            [ew_map, row_map, ew_map],
            [parallel, parallel, parallel],
        )
        def out(o, denom, dst):
            return arith.DivFOp(o, denom)

        bufferization.materialize_in_destination(
            None, out, out_view_memref, restrict=True, writable=True
        )
        self.kinds.append("fa")

    # ---- fused multi-head attention(ln_f32 (n_ctx,n_embd) f32) -> (n_ctx,n_embd) f16, non-causal ----
    def fused_attention(self, x, wq, wk, wv, n_ctx, n_embd, n_head):
        # True multi-head attention via the flash kernel, with no on-device
        # head-transpose kernel. Flow:
        #   x(f32) -cast-> f16 -q/k/v proj-> (n_ctx,n_embd) f16 buffers -heads_view (free)->
        #   (n_head,n_ctx,d_head) strided views -> attention_4d (fused flash kernel) -> @V written
        #   into a (n_ctx,n_embd) f16 buffer via its (n_head,n_ctx,d_head) view -> return that (n_ctx,n_embd) f16.
        d_head = n_embd // n_head
        x16 = self.cast_f16(x, n_ctx, n_embd)  # ew
        qbuf = self.cast_f16_buf(
            self.matmul(x16, wq, n_ctx, n_embd), n_ctx, n_embd
        )  # mm, ew -> (n_ctx,n_embd) f16 memref
        kbuf = self.cast_f16_buf(
            self.matmul(x16, wk, n_ctx, n_embd), n_ctx, n_embd
        )  # mm, ew
        vbuf = self.cast_f16_buf(
            self.matmul(x16, wv, n_ctx, n_embd), n_ctx, n_embd
        )  # mm, ew
        Qh = self.heads_view(
            qbuf, n_ctx, n_head, d_head
        )  # (n_head,n_ctx,d_head) strided view (free)
        Kh = self.heads_view(kbuf, n_ctx, n_head, d_head)
        Vh = self.heads_view(vbuf, n_ctx, n_head, d_head)
        # Output (n_ctx,n_embd) f16 buffer, viewed as (n_head,n_ctx,d_head) for the @V store.
        out_buf = self._buf((n_ctx, n_embd), self.f16)
        out_view_memref = self._heads_view_of(out_buf, n_ctx, n_head, d_head)
        out_view = emit_buf_to_tensor(out_view_memref, restrict=True, writable=True)
        self.attention_4d(
            Qh, Kh, Vh, n_head, n_ctx, d_head, out_view, out_view_memref
        )  # fa, writes out_buf
        return emit_buf_to_tensor(out_buf, restrict=True)  # (n_ctx,n_embd) f16


# ---------------------------------------------------------------------------
# PAYLOAD ASSEMBLY -- wire the Builder ops into a complete MLIR function.
# `build_gpt_fused_payload` creates one `func.func` (the "payload") whose arguments
# are the input + all weights (as device memrefs) and whose body is the op graph.
# `emit_buf_to_tensor` turns a memref argument into a tensor the ops can read;
# `func_cif` makes the function callable from C/the Runner. Returns (module,
# kinds) where `kinds` drives the schedule.
# ---------------------------------------------------------------------------


def _emit_block_fused(bld, x, w, n_ctx, n_embd, d_ffn, n_head, eps, out_buf=None):
    """Emit one transformer block whose attention sublayer is the fused multi-head
    flash kernel (non-causal, no mask). `w` weight keys: g1,b1n, wq,wk,wv, wp,bp,
    g2,b2n, w1,bb1,w2,bb2. wq/wk/wv/wp are full (n_embd,n_embd)."""
    # ---- attention sublayer: a = x + proj(fused_attn(ln1(x))) ----
    ln1 = bld.layernorm(x, w["g1"], w["b1n"], n_ctx, n_embd, eps)
    attn16 = bld.fused_attention(
        ln1, w["wq"], w["wk"], w["wv"], n_ctx, n_embd, n_head
    )  # f16 (n_ctx,n_embd)
    proj = bld.matmul(attn16, w["wp"], n_ctx, n_embd)
    proj = bld.bias(proj, w["bp"], n_ctx, n_embd, relu=False)
    a = bld.add(x, proj, n_ctx, n_embd)
    # ---- FFN sublayer: y = a + ffn(ln2(a)) ----
    ln2 = bld.layernorm(a, w["g2"], w["b2n"], n_ctx, n_embd, eps)
    ln2_16 = bld.cast_f16(ln2, n_ctx, n_embd)
    h = bld.matmul(ln2_16, w["w1"], n_ctx, d_ffn)
    h = bld.bias(h, w["bb1"], n_ctx, d_ffn, relu=True)
    h16 = bld.cast_f16(h, n_ctx, d_ffn)
    o = bld.matmul(h16, w["w2"], n_ctx, n_embd)
    o = bld.bias(o, w["bb2"], n_ctx, n_embd, relu=False)
    return bld.add(a, o, n_ctx, n_embd, out_buf=out_buf)


def build_gpt_fused_payload(
    func_name, n_ctx, n_embd, d_ffn, n_vocab, n_layer, n_head, eps=1e-5
):
    """Full nanoGPT forward as one module, with fused multi-head attention per block.
    Multi-head (n_head heads, flash attention), non-causal (no mask), wq/wk/wv/wp are
    (n_embd,n_embd). Embeddings done host-side. Returns (module, kinds)."""
    f32, f16 = F32(), F16()
    mod = ir.Module.create()
    x_t = ir.MemRefType.get((n_ctx, n_embd), f32)  # input activations (256,256) f32
    g_t = ir.MemRefType.get((n_embd,), f32)  # layernorm gamma/beta vectors (256,) f32
    wqkv_t = ir.MemRefType.get(
        (n_embd, n_embd), f16
    )  # q/k/v projection weights (256,256) f16
    wproj_t = ir.MemRefType.get(
        (n_embd, n_embd), f16
    )  # attention output proj weight (256,256) f16
    bvec_t = ir.MemRefType.get((n_embd,), f32)  # bias vectors (256,) f32
    w1_t = ir.MemRefType.get((n_embd, d_ffn), f16)  # FFN up-projection (256,1024) f16
    b1_t = ir.MemRefType.get((d_ffn,), f32)  # FFN hidden bias (1024,) f32
    w2_t = ir.MemRefType.get((d_ffn, n_embd), f16)  # FFN down-projection (1024,256) f16
    lmw_t = ir.MemRefType.get((n_embd, n_vocab), f16)  # lm_head weight (256,256) f16
    lmb_t = ir.MemRefType.get((n_vocab,), f32)  # lm_head bias (256,) f32
    out_t = ir.MemRefType.get((n_ctx, n_vocab), f32)  # output logits (256,256) f32
    # per-layer arg types: g1,b1n, wq,wk,wv, wp,bp, g2,b2n, w1,bb1,w2,bb2 (13) -- no mask.
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

    fargs = [out_t, x_t]
    for _ in range(n_layer):
        fargs += per_layer
    fargs += [g_t, g_t, lmw_t, lmb_t]  # ln_f gamma/beta, lm_head W,b (no mask)
    bld = Builder()
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
                h = _emit_block_fused(bld, h, w, n_ctx, n_embd, d_ffn, n_head, eps)
            hf = bld.layernorm(h, gf_g, gf_b, n_ctx, n_embd, eps)
            hf16 = bld.cast_f16(hf, n_ctx, n_embd)
            logits = bld.matmul(hf16, lmw, n_ctx, n_vocab)
            bld.bias(logits, lmb, n_ctx, n_vocab, relu=False, out_buf=output)
            for b in bld.to_dealloc:
                gpu.dealloc(None, [], b)

        emit_gpu_util_funcs(f32, rank=2)
        emit_gpu_util_funcs(f32, rank=1)
        emit_gpu_util_funcs(f16, rank=2)
    return mod, bld.kinds

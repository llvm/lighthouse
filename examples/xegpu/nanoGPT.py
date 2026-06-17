# RUN: %PYTHON %s --dump xegpu-wg --gpt-layers 1 | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""nano-GPT / GPT-2-style forward pass on the Intel GPU (XeGPU), with
FLASH multi-head attention.

This is a nanoGPT block stack: each transformer block is
    a = x + attn_proj( MultiHeadAttention( ln1(x) ) )       # attention sublayer
    y = a + ffn( ln2(a) )                                    # MLP sublayer
    ffn(z) = Linear(C, 4C) -> ReLU -> Linear(4C, C)
and the full model is
    x = token_emb + pos_emb            # embeddings (done host-side)
    for _ in range(n_layer): x = Block(x)
    x = ln_f(x); logits = x @ lm_head
TRUE multi-head: H heads of head_size = C/H = 64, computed by ONE
fused flash-attention kernel per block.

The attention is the FLASH/FUSED kernel : standard
attention is built on 4D tensors (Z, H, n_ctx, head_size) at the linalg level,
then a transform-dialect schedule rewrites the whole Q@K^T -> softmax -> @V region
into ONE kernel that tiles the K/V reduction dim and carries a running max/sum (the
flash-attention online-softmax), so the full T x T scores matrix is never
materialized. Everything else (layernorm, the q/k/v/proj/ffn/lm_head matmuls, the
casts/bias/residual elementwise ops) is lowered as its own XeGPU kernel; the whole
model is ONE MLIR module with on-device buffers handing off between kernels.

Config: n_layer=6, C=256, H=4 (head_size=64), hidden=1024, vocab=256, T=256.

Builds the FULL model (n_layer blocks -> ln_f -> lm_head), with FUSED multi-head
NON-CAUSAL attention per block.

Bridging the model's 2D (T,C) activations to the fused kernel's multi-head
(H,T,hs) layout uses NO on-device transpose kernel: each q/k/v projection buffer
is presented as a (H,T,hs) STRIDED memref VIEW (memref.expand_shape +
memref.transpose -- pure layout, zero compute), and the fused schedule's
(1,wg_rows,0,0) tiling peels the head dim into the work-group GRID so each wg reads
2D strided slices -> 2D load_nd .

Run:
  .venv/bin/python examples/xegpu/gpt2.py [--gpt-layers N] [--check]
  .venv/bin/python examples/xegpu/gpt2.py [--dump STAGE]
"""

import sys
import numpy as np
from mlir import ir
from mlir.dialects import linalg, bufferization, tensor, arith, math, gpu, memref
from mlir.dialects import transform
from mlir.dialects.transform import structured, loop, xegpu
from mlir.dialects.transform import bufferization as transform_bufferization
from mlir.dialects.transform.vector import (
    apply_patterns_vector_cast_away_vector_leading_one_dim,
    apply_patterns_vector_drop_unit_dims_with_shape_cast,
)
from mlir.dialects.bufferization import LayoutMapOption

from lighthouse import dialects as lh_dialects
import lighthouse.transform as lh_transform
from lighthouse.dialects.transform import transform_ext
from lighthouse.pipeline.helper import (
    apply_registered_pass,
    canonicalize,
    match,
    match_and_split,
    PipelineInterrupt,
)
from lighthouse.schedule import schedule_boilerplate
from lighthouse.ingress.mlir_gen.utils import (
    emit_buf_to_tensor,
    affine_map,
    parallel,
    reduction,
)
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.ingress.mlir_gen.named import times_weights
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution.runner import Runner
from lighthouse.execution import GPUMemoryManager
from lighthouse.schedule.xegpu import xegpu_to_binary, XeGPUParameterSelector
from lighthouse.schedule.xegpu.mlp_schedule import xegpu_wg_annotation_for_mlp_layer


# =============================================================================
# HOW THIS FILE IS ORGANIZED
# =============================================================================
# Compiling a model to the GPU here happens in THREE stages. The code is grouped
# to match them:
#
#   1. PAYLOAD  -- "WHAT to compute".  Build an MLIR module describing the math
#      as high-level `linalg` ops (matmul, layernorm, softmax, elementwise). This
#      is hardware-agnostic; it does not say how to run on the GPU yet.
#         -> class `Builder` (emits one op at a time) and the `build_*_payload`
#            functions (assemble ops into ffn / attn / block / full-gpt).
#
#   2. SCHEDULE -- "HOW to lower it to the GPU".  Build a SECOND MLIR module (a
#      "transform dialect" script) that rewrites the payload: tile each op into
#      GPU work-groups, vectorize, bufferize, outline each op into its own GPU
#      kernel, and attach XeGPU layout/target attributes.
#         -> `build_combined_schedule` / `_bundle` (the orchestrator) plus the
#            `_tile_one_matmul` / `_tile_one_layernorm` / `_tile_one_fused_attention_region`
#            helpers.
#         *** THIS is the part that "schedules the passes". ***
#
#   3. DRIVER   -- "run it".  `main()` applies the schedule to the payload
#      (TransformDriver), JIT-compiles + runs it on the GPU (Runner), and checks
#      the result against a plain-numpy reference (`numpy_ref_*`).
#
# KEY IDEA -- one module, many separate kernels: the whole model is ONE MLIR
# module, but each op becomes its OWN GPU kernel (no cross-op fusion). Data passes
# between kernels through device buffers (`gpu.alloc`) that stay on the GPU -- no
# round-trip to the host between ops.
# =============================================================================


def F32():  # 32-bit float (used for accumulation / norms)
    return ir.F32Type.get()


def F16():  # 16-bit float (required by the GPU matmul units)
    return ir.F16Type.get()


# =============================================================================
# STAGE 1 -- PAYLOAD: describe WHAT to compute (hardware-agnostic linalg ops)
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

    # ---- softmax(x (M,N) f32) over last dim -> (M,N) f32 buffer ----
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
# Each build_*_payload creates one `func.func` (the "payload") whose arguments
# are the input + all weights (as device memrefs) and whose body is the op graph.
# `emit_buf_to_tensor` turns a memref argument into a tensor the ops can read;
# `func_cif` makes the function callable from C/the Runner. Returns (module,
# kinds) where `kinds` drives the schedule.
# ---------------------------------------------------------------------------


def _mha(q, k, v, H, causal=False):
    """Multi-head attention over (T,C) q/k/v (already projected), per-head, with an
    optional causal mask. Returns (T,C). Mirrors the fused kernel's math, which is
    non-causal, so `causal` defaults to False."""
    T, C = q.shape
    hs = C // H
    scale = 1.0 / (hs**0.5)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1) if causal else 0.0
    attn = np.zeros((T, C), np.float32)
    for h in range(H):
        sl = slice(h * hs, (h + 1) * hs)
        scores = (q[:, sl] @ k[:, sl].T) * scale + mask
        scores = scores - scores.max(-1, keepdims=True)
        e = np.exp(scores)
        w = e / e.sum(-1, keepdims=True)
        attn[:, sl] = _f16(w) @ v[:, sl]
    return attn


def _emit_block_fused(bld, x, w, T, C, hidden, H, eps, out_buf=None):
    """Like _emit_block but the attention sublayer is the FUSED multi-head flash
    kernel (NON-CAUSAL, no mask). `w` weight keys: g1,b1n, wq,wk,wv, wp,bp,
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


def numpy_ref_block_fused(x, w, H, eps=1e-5, causal=False):
    """Multi-head block reference (matches _emit_block_fused)."""
    ln1 = _f16(_ln(x, w["g1"], w["b1n"], eps))
    q = _f16(ln1 @ w["wq"].astype(np.float32))
    k = _f16(ln1 @ w["wk"].astype(np.float32))
    v = _f16(ln1 @ w["wv"].astype(np.float32))
    attn = _mha(q, k, v, H, causal)
    proj = _f16(attn) @ w["wp"].astype(np.float32) + w["bp"]
    a = x + proj
    ln2 = _f16(_ln(a, w["g2"], w["b2n"], eps))
    hh = np.maximum(_f16(ln2) @ w["w1"].astype(np.float32) + w["bb1"], 0.0)
    o = _f16(hh) @ w["w2"].astype(np.float32) + w["bb2"]
    return a + o


def build_gpt_fused_payload(func_name, T, C, hidden, vocab, n_layer, H, eps=1e-5):
    """Full nanoGPT.py forward as ONE module, with FUSED multi-head attention per block
    (R3/R4). Multi-head (H heads, flash attention),
    NON-CAUSAL (no mask), wq/wk/wv/wp are (C,C). Embeddings done host-side."""
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


def numpy_ref_gpt_fused(x, layer_w, gf_g, gf_b, lmw, lmb, H, eps=1e-5):
    """Non-causal multi-head full-gpt reference."""
    h = x
    for w in layer_w:
        h = numpy_ref_block_fused(h, w, H, eps)
    hf = _ln(h, gf_g, gf_b, eps)
    return _f16(hf) @ lmw.astype(np.float32) + lmb


# =============================================================================
# STAGE 2 -- SCHEDULE: describe HOW to lower the payload to the GPU
# =============================================================================
# A "schedule" here is itself an MLIR module written in the TRANSFORM dialect: a
# little program of rewrite ops that the transform interpreter runs over the
# payload module. It does NOT compute anything; it REWRITES the payload from
# high-level linalg ops down to GPU (XeGPU) kernels.
#
# We can't reuse the repo's per-op schedules (layer_norm_schedule, mlp_schedule,
# softmax_schedule) directly, because each assumes the module contains ONLY its
# op. Our module is mixed (matmul + layernorm + softmax + elementwise), so we
# build ONE COMBINED schedule that handles all op classes. The strategy:
#
#   (a) TILE each op into its own parallel loop nest (`scf.forall` = the GPU
#       work-group grid). Different op classes tile differently:
#         - matmul   -> `_tile_one_matmul`  (work-group tile + k-loop tile; the
#                        DPAS tile sizes come from `mm_params`)
#         - layernorm-> `_tile_one_layernorm` (tile rows, fuse the 2 reductions +
#                        2 zero-fills into the loop)
#         - fused attn-> `_tile_one_fused_attention_region` (tile @V batch_matmul into
#                        a forall, fuse QK^T/scale/softmax/@V in; flash rewrite later)
#         - elementwise -> a single `structured_tile_using_forall` over rows
#   (b) SHARED TAIL (same for every kernel): vectorize -> bufferize (tensors ->
#       memrefs) -> convert the forall grids to `gpu.launch` -> OUTLINE each into
#       its own `gpu.module`/`gpu.func` kernel -> attach the XeVM target.
#   (c) ANNOTATE each kernel with XeGPU layout attributes (how data maps to
#       sub-groups / DPAS tiles).
#
# `kinds` (from the Builder) tells the schedule the class and order of every
# kernel, so steps (a) and (c) can treat each one correctly. See memory.md
# parts 6-10 for the subtle correctness rules baked into the helpers below.
# =============================================================================
def _tile_one_matmul(matmul_op, anytype, mm_params):
    """Tile ONE matmul for DPAS: a work-group `forall` tile (wg_m x wg_n) with any
    elementwise consumer fused in, then an inner reduction (k) loop. Tile sizes
    come from `mm_params` (chosen by xegpu_parameter_selector for the GPU)."""
    wg_tile = [mm_params["wg_m"], mm_params["wg_n"]]
    consumers = transform_ext.get_tileable_consumers(matmul_op)
    leaf = transform_ext.extract_handle(consumers, -1)
    _, [wg_loop], _ = lh_transform.tile(
        leaf,
        tile_sizes=wg_tile,
        fuse_producers=True,
        use_forall=True,
        apply_cleanup=False,
    )
    wg_matmul = match(wg_loop, ops={"linalg.matmul"})
    lh_transform.tile(wg_matmul, tile_sizes=[0, 0, mm_params["k_tile"]])


def _tile_one_fused_attention_region(anytype, qkt_bmm, pv_bmm, softmax_op, fa_params):
    """Tile + fuse ONE attention region (QK^T -> scale -> softmax -> @V) into a
    SINGLE scf.forall, so it vectorizes/bufferizes into one kernel body that
    `replace_with_fused_attention` later rewrites into the flash loop.

    Operates on PRE-SPLIT, per-region
    handles (qkt_bmm, pv_bmm, softmax_op) so it is region-local and works at any
    multiplicity. All further producers are pulled in via get_producer_of_operand
    (SSA-walk = inherently scoped to this region)."""
    prod = transform.get_producer_of_operand

    def fuse(p, c):
        return structured.structured_fuse_into_containing_op(
            anytype, anytype, producer_op=p, containing_op=c
        )[1]

    wg_rows = fa_params["wg_rows"]
    # 1. Tile the @V batch_matmul in (batch=1, M=wg_rows) -> forall grid.
    tiled_pv, forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        pv_bmm,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(1, wg_rows, 0, 0),
    )
    func = transform.get_parent_op(
        anytype, forall, op_name="func.func", deduplicate=True
    )
    # 2. Fuse the @V output init fill (producer of forall operand 0).
    forall = fuse(prod(anytype, forall, operand_number=0), forall)
    transform.apply_cse(func)
    canonicalize(func)
    # 3. Decompose this region's softmax. linalg.softmax -> 4 generics + 2 fills:
    #    max = reduce_max(scaled)         [+ -inf fill]
    #    num = exp(scaled - max)
    #    den = reduce_sum(num)            [+ 0 fill]
    #    div = num / den                  (feeds @V)
    structured.structured_decompose_interface(anytype, softmax_op)
    transform.apply_cse(func)
    canonicalize(func)
    # Grab the whole producer chain UP FRONT via SSA walk (region-local; no count
    # matching). Fusing op X invalidates only X's handle, so collect all, then fuse
    # each once in consumer->producer topological order.
    # tiled_pv operand 0 is the aw extract_slice inside the forall; hop through it
    # to the func-scope softmax `div` that it slices.
    aw_slice = prod(anytype, tiled_pv, operand_number=0)
    div = prod(anytype, aw_slice, operand_number=0)  # num / den (softmax out)
    num = prod(anytype, div, operand_number=0)  # exp generic
    den = prod(anytype, div, operand_number=1)  # sum-reduce generic
    den_fill = prod(anytype, den, operand_number=1)  # 0 fill (sum acc)
    mx = prod(anytype, num, operand_number=1)  # max-reduce generic
    mx_fill = prod(anytype, mx, operand_number=1)  # -inf fill (max acc)
    scaled = prod(anytype, num, operand_number=0)  # linalg.mul (qkt*scale)
    scale_fill = prod(anytype, scaled, operand_number=1)  # scale-constant fill
    qkt = prod(anytype, scaled, operand_number=0)  # QK^T batch_matmul
    kt = prod(anytype, qkt, operand_number=1)  # K^T transpose
    qkt_fill = prod(anytype, qkt, operand_number=2)  # 0 fill (qkt acc)
    for p in (
        div,
        den,
        num,
        mx,
        scaled,
        qkt,
        den_fill,
        mx_fill,
        scale_fill,
        qkt_fill,
        kt,
    ):
        forall = fuse(p, forall)
    transform.apply_cse(func)
    canonicalize(func)
    return func, forall


def _fuse_attention_in_region(anytype, forall, fa_params):
    """After the shared bufferize+vectorize, rewrite ONE attention region's
    vector.contract pair (QK^T, @V) into the flash loop via the transform
    op. Scoped to `forall` so counts are exact at any multiplicity."""
    contract_ops = match_and_split(forall, ops={"vector.contract"}, nhandles=2)
    first_contract, second_contract = contract_ops[0], contract_ops[1]
    q_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=0
    )
    k_load = transform.get_producer_of_operand(
        anytype, first_contract, operand_number=1
    )
    v_load = transform.get_producer_of_operand(
        anytype, second_contract, operand_number=1
    )
    mulf_op = match_and_split(forall, ops={"arith.mulf"}, nhandles=1)[0]
    scale = transform.get_producer_of_operand(anytype, mulf_op, operand_number=1)
    # NB: the merged fused-attention op is non-causal only -- there is
    # no `causal` parameter yet, so the model runs as non-causal attention.
    transform_ext.replace_with_fused_attention(
        q_load=q_load,
        k_load=k_load,
        v_load=v_load,
        scale=scale,
        output=second_contract,
        tile_size=fa_params["inner_loop_tile_size"],
    )


def xegpu_fa_annotation(gf, anytype, fa_params):
    """Attach XeGPU layouts to ONE fused-attention gpu.func."""
    num_subgroups = fa_params["wg_rows"] // fa_params["sg_rows"]
    n_head = fa_params["n_head"]
    q_sg_layout = [num_subgroups, 1]
    q_sg_data = [16, n_head]
    q_inst_data = [8, 16]
    k_sg_layout = [num_subgroups, 1]
    k_sg_data = [16, n_head]
    k_inst_data = [16, 16]
    v_sg_layout, v_sg_data, v_inst_data = k_sg_layout, k_sg_data, k_inst_data
    kt_sg_layout = [1, num_subgroups]
    kt_sg_data = [n_head, 16]
    kt_inst_data = [16, 16]
    kt_order = [0, 1]
    out_sg_layout, out_sg_data, out_inst_data = q_sg_layout, q_sg_data, q_inst_data
    l128_sg_layout = [num_subgroups, 1]
    l128_sg_data = [16, 16]
    l128_inst_data = [8, 16]
    qk_sg_layout, qk_sg_data, qk_inst_data = (
        l128_sg_layout,
        l128_sg_data,
        l128_inst_data,
    )

    store_nd_op = match_and_split(gf, ops={"xegpu.store_nd"}, nhandles=1)[0]
    xegpu.set_anchor_layout(
        store_nd_op,
        sg_layout=out_sg_layout,
        sg_data=out_sg_data,
        inst_data=out_inst_data,
    )
    load_nd_ops = match_and_split(gf, ops={"xegpu.load_nd"}, nhandles=9)
    xegpu.set_anchor_layout(
        load_nd_ops[0], sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data
    )
    for i in range(1, 5):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=k_sg_layout,
            sg_data=k_sg_data,
            inst_data=k_inst_data,
        )
    for i in range(5, 9):
        xegpu.set_anchor_layout(
            load_nd_ops[i],
            sg_layout=v_sg_layout,
            sg_data=v_sg_data,
            inst_data=v_inst_data,
        )
    dpas_ops = match_and_split(gf, ops={"xegpu.dpas"}, nhandles=8)
    for i in range(4):
        d = dpas_ops[i]
        xegpu.set_anchor_layout(
            d, sg_layout=q_sg_layout, sg_data=q_sg_data, inst_data=q_inst_data, index=0
        )
        xegpu.set_anchor_layout(
            d,
            sg_layout=kt_sg_layout,
            sg_data=kt_sg_data,
            inst_data=kt_inst_data,
            order=kt_order,
            index=1,
        )
        xegpu.set_anchor_layout(
            d,
            sg_layout=l128_sg_layout,
            sg_data=l128_sg_data,
            inst_data=l128_inst_data,
            index=2,
        )
    for i in range(4, 8):
        d = dpas_ops[i]
        xegpu.set_anchor_layout(
            d,
            sg_layout=qk_sg_layout,
            sg_data=qk_sg_data,
            inst_data=qk_inst_data,
            index=0,
        )
        xegpu.set_anchor_layout(
            d, sg_layout=v_sg_layout, sg_data=v_sg_data, inst_data=v_inst_data, index=1
        )
        xegpu.set_anchor_layout(
            d,
            sg_layout=out_sg_layout,
            sg_data=out_sg_data,
            inst_data=out_inst_data,
            index=2,
        )


def build_combined_schedule(
    mm_params, ln_params, kinds, stop_at_stage="", fa_params=None
):
    """Build the transform-dialect schedule module for a payload with op classes
    `kinds`. Counts how many of each class there are, then delegates to `_bundle`
    (wrapped in transform boilerplate). `stop_at_stage` lets callers halt early
    for debugging (--dump <stage>)."""
    n_mm = kinds.count("mm")
    n_ln = kinds.count("ln")
    n_sm = kinds.count("sm")
    n_ew = kinds.count("ew")
    with schedule_boilerplate() as (schedule, named_seq):
        anytype = transform.AnyOpType.get()
        func0 = match(named_seq.bodyTarget, ops={"func.func"})
        mod = transform.get_parent_op(
            anytype, func0, op_name="builtin.module", deduplicate=True
        )
        try:
            _bundle(
                mod,
                mm_params,
                ln_params,
                kinds,
                n_mm,
                n_ln,
                n_sm,
                n_ew,
                stop_at_stage,
                fa_params=fa_params,
            )
        except PipelineInterrupt:
            pass
        finally:
            transform.yield_()
    return schedule


def _bundle(
    mod,
    mm_params,
    ln_params,
    kinds,
    n_mm,
    n_ln,
    n_sm,
    n_ew,
    stop_at_stage="",
    fa_params=None,
):
    """THE PASS ORCHESTRATOR -- emits the actual sequence of transform ops.

    Runs in 3 phases over the whole payload module:
      TILE   -- tile every op into a GPU work-group `forall` (per op class)
      SHARED TAIL -- vectorize, bufferize, forall->gpu.launch, outline kernels,
                     attach the XeVM target, lower vector ops to XeGPU
      ANNOTATE -- attach XeGPU sub-group/DPAS layout to each kernel
    `stop_at_stage` raises PipelineInterrupt to halt after a phase (for --dump).
    Reading the inline comments here is the best way to understand "which part of
    the code schedules the passes" -- it is this function, top to bottom."""
    anytype = transform.AnyOpType.get()
    rss = ln_params["reduction_step_size"]
    wg_rows = ln_params["wg_rows"]
    nkernels = len(kinds)
    n_fa = kinds.count("fa")

    if stop_at_stage == "initial":
        raise PipelineInterrupt()

    # ===== TILE each op-class into its own forall =====
    # KEY PROBLEM: match(linalg.generic) is NOT scoped -- once an op is tiled into
    # a forall, its generic is STILL matched (it's just nested), so we can't
    # re-match "the remaining bare generics" by count. SOLUTION: split ALL generic
    # handles ONCE up front (their build order is deterministic), then tile each
    # using its preserved handle. A handle to op X stays valid across tiling of
    # OTHER ops. We tile the simple EW generics first (no fusion/cleanup, so ln
    # handles survive), then the layernorms (which fuse + cleanup).
    #
    # Generic build order: each layernorm contributes [mean, var, normalize] (3),
    # in block build order; each elementwise contributes 1. We reconstruct the
    # per-op handle slices from `kinds`.
    # 'fa' softmax generics do NOT exist yet (fa is tiled last, softmax still
    # un-decomposed), so they are not in this pool. The fa core's linalg.transpose
    # /linalg.mul/batch_matmul are not linalg.generic, so also excluded. (The head
    # reshape is a pure memref VIEW -- no generic, no kernel; see Builder.heads_view.)
    ngen_total = 3 * n_ln + n_ew
    gen_handles = transform.split_handle(
        (anytype,) * ngen_total, match(mod, ops={"linalg.generic"})
    )
    # Walk kinds to assign generic handles to ops.
    ln_slices, ew_handles = [], []
    gi = 0
    for k in kinds:
        if k == "ln":
            ln_slices.append(
                (gen_handles[gi], gen_handles[gi + 1], gen_handles[gi + 2])
            )
            gi += 3
        elif k == "ew":
            ew_handles.append(gen_handles[gi])
            gi += 1
        # mm / sm / fa contribute no bare linalg.generic here

    # 1) Tile layernorms FIRST, using preserved (mean,var,normalize) handles.
    #    Doing this BEFORE EW/matmul tiling keeps the bare linalg.fill pool exactly
    #    predictable: 2*(untiled lns) + n_mm (matmul accumulator fills). EW tiling
    #    can introduce its own init fills, so we must finish ln fill-fusion first.
    for i, (mean_red, var_red, normalize) in enumerate(ln_slices):
        ln_untiled = n_ln - i
        _tile_one_layernorm(
            mod,
            anytype,
            wg_rows,
            rss,
            mean_red,
            var_red,
            normalize,
            ln_untiled,
            n_mm,
            ln_params["T"],
        )

    # 2) Tile EW generics into own foralls (handles preserved across ln tiling).
    for eg in ew_handles:
        structured.structured_tile_using_forall(
            anytype,
            anytype,
            eg,
            num_threads=[],
            tile_sizes=[],
            static_tile_sizes=(wg_rows,),
        )

    # 4) Matmuls (their EW producers already wrapped in foralls)
    mms = match_and_split(mod, ops={"linalg.matmul"}, nhandles=n_mm)
    for mm in mms:
        _tile_one_matmul(mm, anytype, mm_params)

    # 5) Fused-attention regions. Done LAST so the generic pre-split above ran while
    #    each fa softmax was still ONE linalg.softmax (its decomposition generics
    #    don't exist yet, so they can't inflate ngen_total). Pre-split the 2*n_fa
    #    batch_matmuls (build order [QK^T, @V] per region) + n_fa softmaxes by count,
    #    then tile+fuse each region into ONE forall (decompose happens in-region).
    if n_fa:
        fa_bmms = match_and_split(mod, ops={"linalg.batch_matmul"}, nhandles=2 * n_fa)
        fa_softmaxes = match_and_split(mod, ops={"linalg.softmax"}, nhandles=n_fa)
        for r in range(n_fa):
            _tile_one_fused_attention_region(
                anytype, fa_bmms[2 * r], fa_bmms[2 * r + 1], fa_softmaxes[r], fa_params
            )

    func = match(mod, ops={"func.func"})
    lh_transform.cleanup(func)
    if stop_at_stage == "tiled":
        raise PipelineInterrupt()

    # ===== SHARED TAIL =====
    func = structured.structured_vectorize_children_and_apply_patterns(
        anytype, func, fold_type_extensions_into_contract=True
    )
    lh_transform.cleanup(func)
    # Fused-attention regions carry a batch-of-1 dim from the (1,wg_rows,0,0) tiling;
    # drop leading unit dims so the QK^T/@V vector.contracts become 2D, as the flash
    # rewrite expects.
    if n_fa:
        with ir.InsertionPoint(transform.apply_patterns(func).patterns):
            apply_patterns_vector_cast_away_vector_leading_one_dim()
            apply_patterns_vector_drop_unit_dims_with_shape_cast()
        transform.apply_cse(func)
        canonicalize(func)
    if stop_at_stage == "vectorized":
        raise PipelineInterrupt()

    mod = apply_registered_pass(mod, "eliminate-empty-tensors")
    mod = transform_bufferization.OneShotBufferizeOp(
        mod,
        allow_return_allocs_from_loops=True,
        bufferize_function_boundaries=True,
        function_boundary_type_conversion=LayoutMapOption.IdentityLayoutMap,
    ).result
    mod = apply_registered_pass(mod, "fold-memref-alias-ops")
    transform.apply_cse(mod)
    canonicalize(mod)

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(
        func,
        "promote-buffers-to-stack",
        options={
            "max-alloc-size-in-bytes": "8192",
            "max-rank-of-allocated-memref": "2",
        },
    )
    if stop_at_stage == "bufferized":
        raise PipelineInterrupt()

    # ===== FUSED-ATTENTION REWRITE (after bufferize+vectorize, before gpu.launch) =====
    # Re-find each attention forall by kinds index (forall IR order == kinds order,
    # the invariant the launch/gpu_mods loops below also rely on) and rewrite its
    # QK^T/@V vector.contract pair into the flash online-softmax loop. Must run
    # BEFORE forall->gpu.launch so the producer-walks for q/k/v loads stay in-region.
    if n_fa:
        all_foralls = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
        for idx, kind in enumerate(kinds):
            if kind == "fa":
                _fuse_attention_in_region(anytype, all_foralls[idx], fa_params)
        func = match(mod, ops={"func.func"})
        transform.apply_cse(func)
        canonicalize(func)
    if stop_at_stage == "inner-tiled":
        raise PipelineInterrupt()

    wg_loops = match_and_split(mod, ops={"scf.forall"}, nhandles=nkernels)
    for wg_loop in wg_loops:
        loop.loop_forall_to_parallel([anytype], wg_loop)
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "gpu-map-parallel-loops")
    func = apply_registered_pass(func, "convert-parallel-loops-to-gpu")
    func = apply_registered_pass(func, "lower-affine")
    transform.apply_cse(func)
    canonicalize(func)

    # launch threads per kernel, in IR (build) order = `kinds`.
    launches = match_and_split(mod, ops={"gpu.launch"}, nhandles=nkernels)
    mm_threads = (
        (mm_params["wg_m"] // mm_params["sg_m"])
        * (mm_params["wg_n"] // mm_params["sg_n"])
        * 16
    )
    sm_threads = (ln_params["wg_rows"] // ln_params["sg_rows"]) * ln_params[
        "subgroup_size"
    ]
    fa_threads = (
        (fa_params["wg_rows"] // fa_params["sg_rows"]) * fa_params["subgroup_size"]
        if fa_params
        else 0
    )
    for launch, kind in zip(launches, kinds):
        nt = {"mm": mm_threads, "fa": fa_threads}.get(kind, sm_threads)
        xegpu.set_gpu_launch_threads(launch, threads=[nt, 1, 1])

    func = apply_registered_pass(func, "lower-affine")
    canonicalize(func)
    func = apply_registered_pass(func, "gpu-launch-sink-index-computations")
    mod = apply_registered_pass(mod, "gpu-kernel-outlining")
    transform.apply_cse(mod)
    if stop_at_stage == "gpu-outlining":
        raise PipelineInterrupt()

    mod = apply_registered_pass(
        mod, "xevm-attach-target", options={"O": "3", "chip": "pvc"}
    )

    # per-gpu.module convert-vector-to-xegpu. ONLY ln/sm need SLM allocas (their
    # cross-lane reductions go through shared local memory -> store_matrix). The
    # ew kernels (cast/bias/residual) are pure row-parallel: forcing their allocas
    # to SLM creates store_matrix paths that fail to lower. So SLM-ify ln/sm only;
    # leave ew (and mm) as store_nd.
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    sg_layout = [ln_params["sg_rows"], 1]
    sg_data = [ln_params["sg_rows"], rss]
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind in ("ln", "sm"):
            allocas = match(gf, ops={"memref.alloca"})
            transform_ext.update_address_space(allocas, address_space=3)
        gf = apply_registered_pass(gf, "convert-vector-to-xegpu")
        transform.apply_cse(gf)
        if kind == "fa":
            # flash kernel carries state in iter_args (no SLM); hoist invariants.
            gf = apply_registered_pass(gf, "loop-invariant-code-motion")
    transform.apply_cse(mod)
    canonicalize(mod)
    if stop_at_stage == "xegpu-initial":
        raise PipelineInterrupt()

    # ===== PER-KERNEL ANNOTATION =====
    #   mm -> full mlp wg annotation
    #   ln -> store_nd (1) + store_matrix (the SLM reduction stores)
    #   sm -> store_nd (1) + store_matrix (4)
    #   ew -> store_nd (1) only (pure row-parallel, no SLM)
    gpu_mods = match_and_split(mod, ops={"gpu.module"}, nhandles=nkernels)
    for gm, kind in zip(gpu_mods, kinds):
        gf = match(gm, ops={"gpu.func"})
        if kind == "mm":
            xegpu_wg_annotation_for_mlp_layer(gf, **mm_params)
        elif kind == "fa":
            xegpu_fa_annotation(gf, anytype, fa_params)
        else:
            # ln/sm/ew: anchor-layout their store_nd, and (ln/sm) their SLM
            # store_matrix. Pass the whole match handle to set_anchor_layout (it
            # accepts a multi-handle) -- avoids guessing exact store counts.
            xegpu.set_anchor_layout(
                match(gf, ops={"xegpu.store_nd"}), sg_layout=sg_layout, sg_data=sg_data
            )
            if kind in ("ln", "sm"):
                xegpu.set_anchor_layout(
                    match(gf, ops={"xegpu.store_matrix"}),
                    sg_layout=sg_layout,
                    sg_data=sg_data,
                )
    if stop_at_stage == "xegpu-wg":
        raise PipelineInterrupt()
    return mod


def _tile_one_layernorm(
    mod, anytype, wg_rows, rss, mean_red, var_red, normalize, ln_untiled, n_mm, T_ROWS
):
    """Tile ONE layernorm into its own forall, using PRESERVED handles to its 3
    generics (mean_red, var_red, normalize). Handles to other ops stay valid.

    The 2 accumulator fills are matched by their producer relationship: we match
    all fills and fuse the ones that feed this ln. To avoid touching matmul fills,
    we rely on fuse_into_containing pulling only genuine producers of the forall.
    """
    _, ln_forall = structured.structured_tile_using_forall(
        anytype,
        anytype,
        normalize,
        num_threads=[],
        tile_sizes=[],
        static_tile_sizes=(wg_rows,),
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=var_red, containing_op=ln_forall
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=mean_red, containing_op=ln_forall
    )
    # Fuse this ln's 2 accumulator fills into the forall. Robustly select ONLY the
    # layernorm accumulator fills (NOT matmul fills) by filtering on result type:
    # ln accumulators are rank-1 tensor<T x f32>; matmul accumulators are rank-2.
    # This avoids fragile positional counting across the whole block. There are
    # 2*ln_untiled such rank-1 fills (this ln + other untiled lns); this ln's are
    # the FIRST 2 in IR order.
    ln_func = transform.get_parent_op(
        anytype, ln_forall, op_name="func.func", deduplicate=True
    )
    reduce_t = ir.RankedTensorType.get((T_ROWS,), F32())  # ln accumulator type (T,)
    fill_match = structured.MatchOp(
        anytype, ln_func, ops=["linalg.fill"], filter_result_type=reduce_t
    )
    n_ln_fills = 2 * ln_untiled
    fills = transform.split_handle((anytype,) * n_ln_fills, fill_match.results[0])
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[1], containing_op=ln_forall
    )
    _, ln_forall = structured.structured_fuse_into_containing_op(
        anytype, anytype, producer_op=fills[0], containing_op=ln_forall
    )
    # Fusion leaves the full-size original fills DEAD at func scope (fusion only
    # slices a copy inside the forall). They must be removed or the next ln finds
    # too many. Use canonicalize (which does DCE of the dead originals) at FUNC
    # scope, but NEVER apply_cse at func scope -- CSE would merge the identical
    # live zero-fills ACROSS layernorms. CSE the duplicate GENERICS inside the
    # forall only (scoped), so the re-match below finds exactly 3.
    transform.apply_cse(ln_forall)
    canonicalize(ln_func)
    # tile this ln's reductions+normalize (now inside the forall). Re-match the
    # 3 generics INSIDE the forall (scoped to ln_forall, so unambiguous: exactly 3).
    g2 = match_and_split(ln_forall, ops={"linalg.generic"}, nhandles=3)
    structured.TileUsingForOp(g2[2], sizes=[0, rss])
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[1], tile_sizes=[0, rss]
    )
    structured.structured_tile_reduction_using_for(
        [anytype], anytype, anytype, anytype, target=g2[0], tile_sizes=[0, rss]
    )
    transform.apply_cse(ln_forall)
    canonicalize(ln_forall)


# =============================================================================
# NUMPY REFERENCE -- the same math in plain numpy, to CHECK the GPU result.
# These mirror what the model computes. `_f16` rounds through float16 to model
# the GPU's f16 matmul precision, so the comparison tolerance can be tight.
# =============================================================================
def _ln(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def _f16(a):
    # round f32 -> f16 -> f32: models the precision loss of the GPU's f16 matmul.
    return a.astype(np.float16).astype(np.float32)


def main():
    """Entry point. Builds the FULL gpt model (n_layer blocks -> ln_f -> lm_head),
    flash multi-head NON-CAUSAL attention per block. Flags:
      --gpt-layers N               : number of transformer layers (default 6)
      --check                      : run on the GPU and compare to the numpy reference
      --dump STAGE                 : print IR at a stage and exit, one of
                                     initial | schedule | tiled | vectorized |
                                     bufferized | inner-tiled | gpu-outlining |
                                     xegpu-initial | xegpu-wg | final

    Flow: build payload module -> build combined schedule (which folds in the fused
    attention rewrite) -> TransformDriver lowers it to XeGPU + xegpu_to_binary makes
    the GPU binary -> Runner JIT-runs it -> compare to the numpy reference."""
    dump = None
    check = "--check" in sys.argv
    if "--dump" in sys.argv:
        dump = sys.argv[sys.argv.index("--dump") + 1]

    # Kernel-friendly shapes: T=C=256 (q/k/v/proj matmuls),
    # hidden=1024, vocab=256, n_layer=6. True multi-head: H heads of
    # head_size=C/H=64 -- the fused flash kernel handles head_size=64 fine.
    T, C, hidden = 256, 256, 1024
    vocab, n_layer = 256, 6
    H = 4  # attention heads (hs = C/H = 64)
    if "--gpt-layers" in sys.argv:
        n_layer = int(sys.argv[sys.argv.index("--gpt-layers") + 1])
    # mm/sm params drive the non-attention kernels (matmul, layernorm); fa_params
    # drives the fused attention kernel.
    param_selector = XeGPUParameterSelector()
    mm_params = dict(param_selector.get_parameters((T, C, C)))
    mm_params["gpu_specs"] = param_selector.gpu_specs
    ln_params = {
        "wg_rows": 64,
        "sg_rows": 8,
        "subgroup_size": 16,
        "reduction_step_size": 16,
        "T": T,
    }
    fa_params = {
        "batch_size": 1,
        "num_heads": H,
        "n_ctx": T,
        "n_head": C // H,
        "wg_rows": 128,
        "sg_rows": 16,
        "subgroup_size": 16,
        "inner_loop_tile_size": 64,
    }

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        mod, kinds = build_gpt_fused_payload("payload", T, C, hidden, vocab, n_layer, H)
        if dump == "initial":
            print(mod)
            print("KINDS:", kinds)
            return

        sched = build_combined_schedule(
            dict(mm_params),
            dict(ln_params),
            kinds,
            stop_at_stage=(dump or ""),
            fa_params=dict(fa_params),
        )
        if dump == "schedule":
            print(sched)
            return
        schedules = [sched]
        if not dump or dump == "final":
            schedules.append(xegpu_to_binary())
        payload = TransformDriver(schedules).apply(mod)
        if dump:
            print(payload)
            return
        print(f"LOWERED OK: 'gpt-fused' to {len(kinds)} kernels in one module")

        if not check:
            return
        runner = Runner(
            payload,
            mem_manager_cls=GPUMemoryManager,
            shared_libs=["libmlir_levelzero_runtime.so"],
        )
        np.random.seed(0)
        out = np.zeros((T, vocab), np.float32)
        cb = Runner.get_gpu_argument_access_callback(out, arg_index=0)
        sc = 0.05  # small weight scale -> O(1) activations so f16 stays accurate

        # full model, fused multi-head attn per block.
        # host "embeddings": simulate token+pos embedding sum as the input x.
        x = (np.random.randn(T, C) * 0.5).astype(np.float32)
        layers = []
        host = [out, x]
        for _ in range(n_layer):
            lw = dict(
                g1=np.ones(C, np.float32),
                b1n=np.zeros(C, np.float32),
                wq=(np.random.randn(C, C) * sc).astype(np.float16),
                wk=(np.random.randn(C, C) * sc).astype(np.float16),
                wv=(np.random.randn(C, C) * sc).astype(np.float16),
                wp=(np.random.randn(C, C) * sc).astype(np.float16),
                bp=np.zeros(C, np.float32),
                g2=np.ones(C, np.float32),
                b2n=np.zeros(C, np.float32),
                w1=(np.random.randn(C, hidden) * sc).astype(np.float16),
                bb1=np.zeros(hidden, np.float32),
                w2=(np.random.randn(hidden, C) * sc).astype(np.float16),
                bb2=np.zeros(C, np.float32),
            )
            layers.append(lw)
            host += [
                lw["g1"],
                lw["b1n"],
                lw["wq"],
                lw["wk"],
                lw["wv"],
                lw["wp"],
                lw["bp"],
                lw["g2"],
                lw["b2n"],
                lw["w1"],
                lw["bb1"],
                lw["w2"],
                lw["bb2"],
            ]
        gf_g = np.ones(C, np.float32)
        gf_b = np.zeros(C, np.float32)
        lmw = (np.random.randn(C, vocab) * sc).astype(np.float16)
        lmb = np.zeros(vocab, np.float32)
        host += [gf_g, gf_b, lmw, lmb]
        runner.execute(
            host_input_buffers=host,
            payload_function_name="payload",
            argument_access_callback=cb,
        )
        ref = numpy_ref_gpt_fused(x, layers, gf_g, gf_b, lmw, lmb, H)

        rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
        print(f"max abs diff={np.abs(out - ref).max():.4f}  rel={rel:.6f}")
        print("PASSED" if rel < 5e-2 else "FAILED")


if __name__ == "__main__":
    main()

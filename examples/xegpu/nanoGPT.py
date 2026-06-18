# RUN: %PYTHON %s --dump xegpu-wg --gpt-layers 1 | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""nano-GPT / GPT-2-style forward pass on the Intel GPU (XeGPU), with
FLASH multi-head attention -- the DRIVER ("run it") entry point.

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

HOW THIS EXAMPLE IS ORGANIZED -- compiling a model to the GPU here happens in
THREE stages:

  1. PAYLOAD  ("WHAT to compute") -> lighthouse/ingress/mlir_gen/gpu_nanoGPT_payload.py
     (the `Builder` class + `build_gpt_fused_payload`). Hardware-agnostic linalg.
  2. SCHEDULE ("HOW to lower it") -> lighthouse/schedule/xegpu/nanoGPT_schedule.py
     (`build_combined_schedule`). A transform-dialect module that tiles each op
     into GPU work-groups, vectorizes, bufferizes, outlines each op into its own
     GPU kernel, and attaches XeGPU layout/target attributes.
  3. DRIVER   ("run it") -> this file. `main()` applies the schedule to the payload
     (TransformDriver), JIT-compiles + runs it on the GPU (Runner), and checks the
     result against the plain-numpy reference below.

KEY IDEA -- one module, many separate kernels: the whole model is ONE MLIR module,
but each op becomes its OWN GPU kernel (no cross-op fusion). Data passes between
kernels through device buffers (`gpu.alloc`) that stay on the GPU -- no round-trip
to the host between ops.

Run:
  .venv/bin/python examples/xegpu/nanoGPT.py [--gpt-layers N] [--check]
  .venv/bin/python examples/xegpu/nanoGPT.py [--dump STAGE]
"""

import sys
import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution.runner import Runner
from lighthouse.execution import GPUMemoryManager
from lighthouse.schedule.xegpu import xegpu_to_binary, XeGPUParameterSelector
from lighthouse.ingress.mlir_gen.gpu_nanoGPT_payload import build_gpt_fused_payload
from lighthouse.ingress.mlir_gen.gpu_utils import emit_gpu_util_funcs
from lighthouse.schedule.xegpu.nanoGPT_schedule import build_combined_schedule


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


def numpy_ref_block_fused(x, w, H, eps=1e-5, causal=False):
    """Multi-head block reference (matches _emit_block_fused in the payload)."""
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


def numpy_ref_gpt_fused(x, layer_w, gf_g, gf_b, lmw, lmb, H, eps=1e-5):
    """Non-causal multi-head full-gpt reference (matches build_gpt_fused_payload)."""
    h = x
    for w in layer_w:
        h = numpy_ref_block_fused(h, w, H, eps)
    hf = _ln(h, gf_g, gf_b, eps)
    return _f16(hf) @ lmw.astype(np.float32) + lmb


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
        with ir.InsertionPoint(mod.body):
            f32, f16 = ir.F32Type.get(), ir.F16Type.get()
            emit_gpu_util_funcs(f32, rank=2)
            emit_gpu_util_funcs(f32, rank=1)
            emit_gpu_util_funcs(f16, rank=2)
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

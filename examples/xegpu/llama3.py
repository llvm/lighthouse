# RUN: %PYTHON %s --dump xegpu-wg --n-layers 1 | FileCheck %s
# CHECK: module attributes {gpu.container_module} {

"""Llama-3 forward pass on the Intel GPU (XeGPU) -- the DRIVER ("run it").

Runs a Llama-3 transformer forward/inference as one MLIR module lowered to many
separate un-fused XeGPU kernels with on-device handoff. Llama building blocks:

  * RMSNorm
  * SwiGLU FFN     w2( silu(z @ w1) * (z @ w3) )
  * grouped-query attention via the fused flash kernel (causal by default;
    --no-causal to disable), with RoPE applied to q/k
  * no biases (Llama uses bias=False on every Linear)

Each transformer block is
    h   = x + wo( MHA( rms_attn(x) ) )       # attention sublayer
    out = h + swiglu( rms_ffn(h) )           # MLP sublayer
and the full model is
    x = tok_embeddings(tokens)         # embeddings done host-side
    for _ in range(n_layers): x = Block(x)
    x = rms_final(x); logits = x @ output_weight

Config: n_layers=6, dim=256, n_heads=4 (head_dim=64), hidden=1024, vocab=256,
seq_len=256.

Three-stage organization (compiling a model to the GPU here):
  1. Payload  ("what to compute") -> examples/xegpu/llama3_payload.py
  2. Schedule ("how to lower it")  -> examples/xegpu/llama3_schedule.py
  3. Driver   ("run it")           -> this file.

Run:
  .venv/bin/python examples/xegpu/llama3.py [--n-layers N] [--check]
  .venv/bin/python examples/xegpu/llama3.py [--dump STAGE]
"""

import argparse
import numpy as np
from mlir import ir

from lighthouse import dialects as lh_dialects
from lighthouse.pipeline.driver import TransformDriver
from lighthouse.execution.runner import Runner
from lighthouse.execution import GPUMemoryManager
from lighthouse.schedule.xegpu import xegpu_to_binary, XeGPUParameterSelector
from llama3_payload import build_llama_payload
from llama3_schedule import build_combined_schedule


# =============================================================================
# NUMPY REFERENCE -- the same math in plain numpy, to CHECK the GPU result.
# `_f16` rounds through float16 to model the GPU's f16 matmul precision.
# =============================================================================
def _rms(x, weight, eps=1e-5):
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


def _f16(a):
    # round f32 -> f16 -> f32: models the precision loss of the GPU's f16 matmul.
    return a.astype(np.float16).astype(np.float32)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rope_tables(T, hs, theta=10000.0):
    """Precompute (cos, sin) tables of shape (T, hs/2) for half-split RoPE.
    freq[i] = theta**(-2i/hs); angle[t,i] = t * freq[i]. Matches the payload's
    cos/sin memref args (see Builder.rope)."""
    half = hs // 2
    freqs = theta ** (-np.arange(0, half, dtype=np.float32) * 2.0 / hs)
    ang = np.outer(np.arange(T, dtype=np.float32), freqs)  # (T, half)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _rope(x, cos, sin, nh):
    """Half-split (GPT-NeoX / HF-Llama) rotary embedding on (T, nh*hs) f32, per head.
    Mirrors Builder.rope: within each head, coord d (first half) pairs with d+half
    (second half): out[d]=a*cos-b*sin, out[d+half]=b*cos+a*sin. cos/sin are (T,half)."""
    T, D = x.shape
    hs = D // nh
    half = hs // 2
    v = x.reshape(T, nh, hs)
    a = v[:, :, :half]
    b = v[:, :, half:]
    c = cos[:, None, :]  # (T,1,half) broadcast over heads
    s = sin[:, None, :]
    out = np.empty_like(v)
    out[:, :, :half] = a * c - b * s
    out[:, :, half:] = b * c + a * s
    return out.reshape(T, D)


def _mha(q, k, v, H, n_kv, causal=False):
    """Grouped-query attention over q (T,C) and narrow k/v (T, n_kv*hs), per query
    head, with an optional causal mask. Returns (T,C). Query head h reads KV head
    h // n_rep (n_rep = H // n_kv), matching the fused kernel's floordiv K/V index.
    Mirrors the fused kernel's math; pass `causal` to match `fa_params["causal"]`."""
    T, C = q.shape
    hs = C // H
    n_rep = H // n_kv
    scale = 1.0 / (hs**0.5)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), k=1) if causal else 0.0
    attn = np.zeros((T, C), np.float32)
    for h in range(H):
        q_sl = slice(h * hs, (h + 1) * hs)
        kv = h // n_rep  # GQA: this query head's KV head
        kv_sl = slice(kv * hs, (kv + 1) * hs)
        scores = (q[:, q_sl] @ k[:, kv_sl].T) * scale + mask
        scores = scores - scores.max(-1, keepdims=True)
        e = np.exp(scores)
        w = e / e.sum(-1, keepdims=True)
        attn[:, q_sl] = _f16(w) @ v[:, kv_sl]
    return attn


def numpy_ref_block_llama(x, w, cos, sin, H, n_kv, eps=1e-5, causal=False):
    """Grouped-query + RoPE Llama block reference (matches _emit_block_llama). wk/wv
    are narrow (C, n_kv*hs), so k/v are (T, n_kv*hs) and _mha does the head grouping.
    RoPE is applied to q and k on the f32 projection, before the f16 cast (v skips
    RoPE) -- matching Builder.fused_attention."""
    rms1 = _f16(_rms(x, w["an"], eps))
    q = _f16(_rope(rms1 @ w["wq"].astype(np.float32), cos, sin, H))
    k = _f16(_rope(rms1 @ w["wk"].astype(np.float32), cos, sin, n_kv))
    v = _f16(rms1 @ w["wv"].astype(np.float32))
    attn = _mha(q, k, v, H, n_kv, causal)
    proj = _f16(attn) @ w["wo"].astype(np.float32)
    h = x + proj
    rms2 = _f16(_rms(h, w["fn"], eps))
    gate = _silu(rms2 @ w["w1"].astype(np.float32))
    up = rms2 @ w["w3"].astype(np.float32)
    o = _f16(gate * up) @ w["w2"].astype(np.float32)
    return h + o


def numpy_ref_llama(x, layer_w, fn_w, lmw, cos, sin, H, n_kv, eps=1e-5, causal=False):
    """Grouped-query + RoPE full-Llama reference (matches build_llama_payload);
    `causal` toggles the autoregressive attention mask to match the fused kernel."""
    h = x
    for w in layer_w:
        h = numpy_ref_block_llama(h, w, cos, sin, H, n_kv, eps, causal)
    hf = _rms(h, fn_w, eps)
    return _f16(hf) @ lmw.astype(np.float32)


def main():
    """Entry point. Builds the full Llama model (n_layers blocks -> rms_final ->
    output), with flash grouped-query attention (RoPE, causal) per block.

    Flow: build payload module -> build combined schedule (which folds in the
    fused attention rewrite) -> TransformDriver lowers it to XeGPU + xegpu_to_binary
    makes the GPU binary -> Runner JIT-runs it -> compare to the numpy reference.
    """
    parser = argparse.ArgumentParser(
        description="Llama-3-style forward pass on the Intel GPU (XeGPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Number of transformer layers (the full model uses 6).",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Disable causal masking (run non-causal attention).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run on the GPU and compare the result against the numpy reference.",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        choices=[
            "initial",
            "schedule",
            "tiled",
            "vectorized",
            "bufferized",
            "inner-tiled",
            "gpu-outlining",
            "xegpu-initial",
            "xegpu-wg",
            "final",
        ],
        help="Print the IR at the given stage and exit.",
    )
    args = parser.parse_args()
    dump = args.dump
    check = args.check
    n_layers = args.n_layers
    causal = not args.no_causal  # Llama-3 is autoregressive/causal

    # Kernel-friendly shapes: T=dim=256 (q/k/v/proj matmuls), hidden=1024,
    # vocab=256. Multi-head: H heads of head_size=dim/H=64. GQA: n_kv KV heads
    # (n_kv <= H, H % n_kv == 0); each query head reads KV head h // (H//n_kv).
    T, C, hidden = 256, 256, 1024
    vocab = 256
    H = 4  # attention (query) heads (hs = C/H = 64)
    # KV heads for grouped-query attention: 1 < n_kv < H, and n_kv | H. Each query
    # head reads KV head h // (H//n_kv). The degenerate ends -- n_kv == H (plain MHA,
    # n_rep=1) and n_kv == 1 (multi-query) -- collapse a factored head dim to 1,
    # which reshapes the fused-attention region so its QK^T/@V don't co-tile into one
    # forall; use the plain-MHA example for n_kv == H.
    n_kv = 2  # 2 query heads share each KV head (n_rep = H // n_kv = 2)
    hs = C // H
    kv_dim = n_kv * hs  # narrow K/V feature width
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
        "causal": causal,
    }

    with ir.Context(), ir.Location.unknown():
        lh_dialects.register_and_load()
        mod, kinds, mm_shapes = build_llama_payload(
            "payload", T, C, hidden, vocab, n_layers, H, n_kv
        )
        if dump == "initial":
            print(mod)
            print("KINDS:", kinds)
            return

        # Per-matmul DPAS params: the K/V projections are narrow (N = n_kv*hs < C),
        # so they need different wg_n/sg_n tiling than the wide matmuls. Select once
        # per distinct (M,N,K) shape (the selector reads the tuple as (M,N,K)).
        shape_params = {}
        for shp in mm_shapes:
            if shp not in shape_params:
                p = dict(param_selector.get_parameters(shp))
                p["gpu_specs"] = param_selector.gpu_specs
                shape_params[shp] = p
        mm_params_list = [dict(shape_params[shp]) for shp in mm_shapes]

        sched = build_combined_schedule(
            dict(mm_params),
            dict(ln_params),
            kinds,
            stop_at_stage=(dump or ""),
            fa_params=dict(fa_params),
            mm_params_list=mm_params_list,
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
        print(f"LOWERED OK: 'llama' to {len(kinds)} kernels in one module")

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

        # host "embeddings": simulate tok_embeddings(tokens) as the input x.
        x = (np.random.randn(T, C) * 0.5).astype(np.float32)
        cos, sin = _rope_tables(T, hs)  # RoPE (T, hs/2) tables, shared across layers
        layers = []
        host = [out, x, cos, sin]  # matches payload arg order: out, x, cos, sin, ...
        for _ in range(n_layers):
            lw = dict(
                an=np.ones(C, np.float32),
                wq=(np.random.randn(C, C) * sc).astype(np.float16),
                wk=(np.random.randn(C, kv_dim) * sc).astype(np.float16),
                wv=(np.random.randn(C, kv_dim) * sc).astype(np.float16),
                wo=(np.random.randn(C, C) * sc).astype(np.float16),
                fn=np.ones(C, np.float32),
                w1=(np.random.randn(C, hidden) * sc).astype(np.float16),
                w2=(np.random.randn(hidden, C) * sc).astype(np.float16),
                w3=(np.random.randn(C, hidden) * sc).astype(np.float16),
            )
            layers.append(lw)
            host += [
                lw["an"],
                lw["wq"],
                lw["wk"],
                lw["wv"],
                lw["wo"],
                lw["fn"],
                lw["w1"],
                lw["w2"],
                lw["w3"],
            ]
        fn_w = np.ones(C, np.float32)
        lmw = (np.random.randn(C, vocab) * sc).astype(np.float16)
        host += [fn_w, lmw]
        runner.execute(
            host_input_buffers=host,
            payload_function_name="payload",
            argument_access_callback=cb,
        )
        ref = numpy_ref_llama(x, layers, fn_w, lmw, cos, sin, H, n_kv, causal=causal)

        rel = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)
        print(f"max abs diff={np.abs(out - ref).max():.4f}  rel={rel:.6f}")
        print("PASSED" if rel < 5e-2 else "FAILED")


if __name__ == "__main__":
    main()

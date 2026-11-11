# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc --implicit-check-not=math.log
# RUN: %PYTHON %S/kernel_bench.py --input %s --input-shapes 1x512x2048 512x64 512x64 --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc --implicit-check-not=math.log
# REQUIRES: torch, torch_mlir

"""Complete cache-free Llama 3.2 1B decoder layer for fixed-shape prefill.

The common XeGPU pipeline composes shared input RMSNorm, Q/K/V projections,
RoPE, causal GQA, output projection/residual, and post-attention RMSNorm plus
SwiGLU/residual. Weights are random; there is no KV cache, padding mask or dropout.
The default schedule uses twelve kernels and nine intermediate allocations.
Head merging and both residual additions are fused, without scratch copies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_rope import get_rope_tables, rotate_half

batch_size = 4
n_ctx = 512
dim = 2048
hidden_dim = 8192
num_heads = 32
num_kv_heads = 8
d_head = 64


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + 1e-5)).to(input_dtype)


class Model(nn.Module):
    """Cache-free Llama 3.2 1B decoder layer for fixed-shape prefill.

    Inputs are hidden states [batch, sequence, 2048] and precomputed cos/sin
    tables [sequence, 64], shared across the batch. Output has the hidden-state
    shape. All projections are bias-free, with randomly initialized weights.
    RoPE table generation is outside the imported graph, as in HuggingFace's
    decoder layer. There is no padding mask, KV cache, or dropout.
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm()
        self.q_proj = nn.Linear(dim, num_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(num_heads * d_head, dim, bias=False)
        self.post_attention_layernorm = RMSNorm()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_proj(h).view(batch, sequence, num_heads, d_head).transpose(1, 2)
        k = self.k_proj(h).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        v = self.v_proj(h).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)

        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rows = torch.arange(sequence, device=x.device)
        allowed = rows[:, None] >= rows[None, :]
        # Avoid torch-mlir's SDPA XPU mask-factory and boolean-mask import issues.
        zero = torch.tensor(0.0, dtype=x.dtype)
        neg_inf = torch.tensor(float("-inf"), dtype=x.dtype)
        mask = torch.where(allowed, zero, neg_inf)
        attention = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True, dropout_p=0.0
        )
        attention = attention.transpose(1, 2).reshape(batch, sequence, dim)
        h = x + self.o_proj(attention)
        normalized = self.post_attention_layernorm(h)
        return h + self.down_proj(
            F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)
        )


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []


# CHECK-COUNT-9: gpu.alloc
# CHECK-COUNT-6: gpu.func @main_kernel
# CHECK: gpu.func @main_kernel
# CHECK: arith.cmpi sge
# CHECK-NEXT: arith.select
# CHECK-NOT: math.log
# CHECK: xegpu.dpas
# CHECK: math.exp
# CHECK: xegpu.dpas
# CHECK: xegpu.store_nd
# CHECK: gpu.func @main_kernel
# CHECK: xegpu.dpas
# CHECK: arith.addf
# CHECK: xegpu.store_nd
# CHECK: gpu.func @main_kernel
# CHECK: vector.multi_reduction
# CHECK: math.rsqrt
# CHECK-NOT: xegpu.dpas
# CHECK: gpu.func @main_kernel
# CHECK: xegpu.dpas
# CHECK: math.exp
# CHECK: xegpu.store_nd
# CHECK: gpu.func @main_kernel
# CHECK: xegpu.dpas
# CHECK: arith.mulf
# CHECK: xegpu.store_nd
# CHECK: gpu.func @main_kernel
# CHECK: xegpu.dpas
# CHECK: arith.addf
# CHECK: xegpu.store_nd
# CHECK-NOT: gpu.func

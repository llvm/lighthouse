# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s
# REQUIRES: torch, torch_mlir

"""Complete cache-free Llama 3.1 8B decoder layer for fixed-shape prefill.

Weights are random; there is no KV cache, padding mask or dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from llama_rope import get_rope_tables, rotate_half

batch_size = 4
n_ctx = 512
dim = 4096
hidden_dim = 14336
num_heads = 32
num_kv_heads = 8
d_head = 128


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
    """Llama 3.1 8B decoder layer with external RoPE tables.

    Inputs are hidden states [batch, sequence, 4096] and cos/sin tables
    [sequence, 128], shared across the batch. All projections are bias-free.
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
    cos, sin = get_rope_tables(torch.arange(n_ctx), d_head, scaling_factor=8.0)
    return [x, cos, sin]


def get_init_inputs():
    return []


# CHECK: gpu.func @main_kernel({{.*}}memref<2048x4096xbf16>{{.*}}memref<512x128xbf16>{{.*}}memref<4x8x4x512x128xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<2048x1024xbf16>{{.*}}memref<512x128xbf16>{{.*}}memref<4x8x512x128xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<4x8x4x512x128xbf16>{{.*}}memref<4x8x512x128xbf16>
# CHECK: xegpu.dpas {{.*}} vector<128x128xbf16>, vector<128x32xbf16>
# CHECK: xegpu.dpas {{.*}} vector<128x32xbf16>, vector<32x128xbf16>
# CHECK: xegpu.store_nd {{.*}} vector<128x128xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<4096x4096xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<14336x4096xbf16>{{.*}}memref<4x512x14336xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<14336x4096xbf16>{{.*}}memref<4x512x14336xbf16>
# CHECK: gpu.func @main_kernel({{.*}}memref<4096x14336xbf16>{{.*}}memref<4x512x4096xbf16>

# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=math.log
# REQUIRES: torch, torch_mlir

"""Llama 3.2 1B prefill attention block, including output projection/residual.

Input RMSNorm is shared by Q/K/V projections. Q/K use half-split RoPE and
causal grouped-query attention. Heads are merged before the bias-free output
projection, and the original unnormalized hidden states supply the residual.
There is no KV cache, padding mask, dropout, or feed-forward block.

The schedule uses eight GPU kernels. Head merging is fused into the attention
store and the residual into the output projection. Seven intermediate tensors
need six allocations because the dead Q-projection buffer is reused.
"""

import torch
import torch.nn as nn

from llama_rope import get_rope_tables, rotate_half


batch_size = 4
n_ctx = 512
dim = 2048
num_heads = 32
num_kv_heads = 8
d_head = 64


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.q_proj = nn.Linear(dim, num_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(num_heads * d_head, dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        h = x.float()
        variance = h.pow(2).mean(dim=-1, keepdim=True)
        h = self.norm_weight * (h * torch.rsqrt(variance + 1e-5)).to(x.dtype)
        q = self.q_proj(h).view(batch, sequence, num_heads, d_head).transpose(1, 2)
        k = self.k_proj(h).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        v = self.v_proj(h).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rows = torch.arange(sequence, device=x.device)
        allowed = rows[:, None] >= rows[None, :]
        # Avoid SDPA's XPU mask-factory and boolean-mask import failures.
        zero = torch.tensor(0.0, dtype=x.dtype)
        neg_inf = torch.tensor(float("-inf"), dtype=x.dtype)
        mask = torch.where(allowed, zero, neg_inf)
        attention = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True
        )
        attention = attention.transpose(1, 2).reshape(
            batch, sequence, num_heads * d_head
        )
        return x + self.o_proj(attention)


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []


# CHECK-COUNT-6: gpu.alloc
# CHECK-NOT: gpu.alloc
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
# CHECK-NOT: gpu.func

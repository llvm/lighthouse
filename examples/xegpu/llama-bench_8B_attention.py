# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=math.log
# REQUIRES: torch, torch_mlir

"""Llama 3.1 8B causal grouped-query attention with pre-projected Q/K/V.

RoPE is not applied here; inputs are assumed to be already projected and
rotated. The additive causal mask follows the 3B attention benchmark.
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        rows = torch.arange(Q.shape[-2], device=Q.device)
        cols = torch.arange(K.shape[-2], device=K.device)
        allowed = rows[:, None] >= cols[None, :]
        zero = torch.tensor(0.0, dtype=Q.dtype)
        neg_inf = torch.tensor(float("-inf"), dtype=Q.dtype)
        mask = torch.where(allowed, zero, neg_inf)
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, is_causal=False, enable_gqa=True
        )


batch_size = 4
n_ctx = 512
num_heads = 32
num_kv_heads = 8
d_head = 128


def get_inputs():
    Q = torch.rand(batch_size, num_heads, n_ctx, d_head)
    K = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    V = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    return [Q, K, V]


def get_init_inputs():
    return []


# CHECK: gpu.func @main_kernel
# CHECK: xegpu.prefetch_nd
# CHECK-SAME: sg_layout = [4, 8], sg_data = [8, 16], inst_data = [8, 16]
# CHECK-SAME: !xegpu.tensor_desc<32x128xbf16
# CHECK: xegpu.dpas {{.*}} vector<128x128xbf16>, vector<128x32xbf16>
# CHECK: xegpu.dpas {{.*}} vector<128x32xbf16>, vector<32x128xbf16>

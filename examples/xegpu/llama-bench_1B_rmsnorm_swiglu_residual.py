# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

"""Llama 3.2 1B post-attention RMSNorm, SwiGLU FFN and residual connection.

One normalization result feeds gate and up projections. Four kernels materialize
normalized input, activated gate and gated activations. The residual add is
fused into down projection's epilogue, using the original unnormalized input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 4
n_ctx = 512
dim = 2048
hidden_dim = 8192


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        variance = h.pow(2).mean(dim=-1, keepdim=True)
        h = self.norm_weight * (h * torch.rsqrt(variance + 1e-5)).to(x.dtype)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


def get_inputs():
    return [torch.randn(batch_size, n_ctx, dim)]


def get_init_inputs():
    return []


# CHECK-COUNT-3: gpu.alloc
# CHECK: gpu.func @main_kernel
# CHECK: vector.multi_reduction
# CHECK: math.rsqrt
# CHECK-NOT: xegpu.dpas
# CHECK: gpu.func @main_kernel
# CHECK: xegpu.dpas
# CHECK: math.exp
# CHECK-NEXT: arith.addf
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

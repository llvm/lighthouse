# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

"""Llama 3.2 1B bias-free SwiGLU FFN, without normalization or residual.

The three GEMM kernels materialize activated gate and gated activations.
SiLU is fused into gate projection; the multiply into up projection's
epilogue. Down projection consumes the gated tensor without recomputing it.
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
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(batch_size, n_ctx, dim)]


def get_init_inputs():
    return []


# CHECK-COUNT-2: gpu.alloc
# CHECK-NOT: gpu.alloc
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
# CHECK: xegpu.store_nd
# CHECK-NOT: gpu.func

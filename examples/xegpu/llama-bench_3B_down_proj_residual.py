# RUN: %PYTHON %S/kernel_bench.py --input %s --disable-tuning --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

"""Llama 3.2 3B FFN down projection with a fused residual addition.

Inputs are gated activations [4, 512, 8192] and the post-attention residual
[4, 512, 3072]. The bias-free projection produces one [4, 512, 3072] tensor.
The flattened GEMM has M=2048, N=3072, K=8192, matching the full decoder.
"""

import torch
import torch.nn as nn


batch_size = 4
n_ctx = 512
dim = 3072
hidden_dim = 8192


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, gated: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + self.down_proj(gated)


def get_inputs():
    return [
        torch.randn(batch_size, n_ctx, hidden_dim),
        torch.randn(batch_size, n_ctx, dim),
    ]


def get_init_inputs():
    return []


# CHECK: gpu.launch_func
# CHECK-NOT: gpu.launch_func
# CHECK: gpu.func @main_kernel
# CHECK-SAME: memref<3072x8192xbf16>
# CHECK-SAME: memref<4x512x8192xbf16>
# CHECK-SAME: memref<4x512x3072xbf16>
# CHECK-SAME: memref<4x512x3072xbf16>
# CHECK: xegpu.dpas
# CHECK: %[[SUM:.*]] = arith.addf
# CHECK-NEXT: {{.*}}memref.subview
# CHECK-NEXT: {{.*}}xegpu.create_nd_tdesc
# CHECK-NEXT: xegpu.store_nd %[[SUM]],
# CHECK-NOT: gpu.func

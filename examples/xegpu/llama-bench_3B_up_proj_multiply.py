# RUN: %PYTHON %S/kernel_bench.py --input %s --disable-tuning --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

"""Llama 3.2 3B FFN up projection with a fused SwiGLU multiply epilogue.

Inputs are normalized hidden states [4, 512, 3072] and a precomputed activated
gate [4, 512, 8192]. The bias-free projection and elementwise multiply produce
one gated activation tensor [4, 512, 8192], without recomputing the gate.
The flattened GEMM has M=2048, N=8192, K=3072, matching the full decoder.
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
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return gate * self.up_proj(x)


def get_inputs():
    return [
        torch.randn(batch_size, n_ctx, dim),
        torch.randn(batch_size, n_ctx, hidden_dim),
    ]


def get_init_inputs():
    return []


# CHECK: gpu.launch_func
# CHECK-NOT: gpu.launch_func
# CHECK: gpu.func @main_kernel
# CHECK-SAME: memref<8192x3072xbf16>
# CHECK-SAME: memref<4x512x3072xbf16>
# CHECK-SAME: memref<4x512x8192xbf16>
# CHECK-SAME: memref<4x512x8192xbf16>
# CHECK: xegpu.dpas
# CHECK: %[[PRODUCT:.*]] = arith.mulf
# CHECK-NEXT: {{.*}}memref.subview
# CHECK-NEXT: {{.*}}xegpu.create_nd_tdesc
# CHECK-NEXT: xegpu.store_nd %[[PRODUCT]],
# CHECK-NOT: gpu.func

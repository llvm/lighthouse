# RUN: %PYTHON %S/kernel_bench.py --input %s --disable-tuning --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc
# REQUIRES: torch, torch_mlir

"""Llama 3.1 8B FFN gate projection with a fused SiLU epilogue.

Input is normalized hidden states [4, 512, 4096]. The bias-free projection
and SiLU produce one activated gate tensor [4, 512, 14336].
The flattened GEMM has M=2048, N=14336, K=4096, matching the full decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 4
n_ctx = 512
dim = 4096
hidden_dim = 14336


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x))


def get_inputs():
    return [torch.randn(batch_size, n_ctx, dim)]


def get_init_inputs():
    return []


# CHECK: gpu.launch_func
# CHECK-NOT: gpu.launch_func
# CHECK: gpu.func @main_kernel
# CHECK-SAME: memref<14336x4096xbf16>
# CHECK-SAME: memref<4x512x4096xbf16>
# CHECK-SAME: memref<4x512x14336xbf16>
# CHECK: xegpu.dpas
# CHECK: math.exp
# CHECK-NEXT: {{.*}}arith.addf
# CHECK-NEXT: {{.*}}arith.divf
# CHECK-NEXT: {{.*}}arith.mulf
# CHECK: xegpu.store_nd
# CHECK-NOT: gpu.func

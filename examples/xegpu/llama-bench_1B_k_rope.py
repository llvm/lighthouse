# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=vector.extract_strided_slice --implicit-check-not=vector.insert_strided_slice --implicit-check-not=memref.copy
# REQUIRES: torch, torch_mlir

# CHECK: xegpu.dpas
# CHECK-LABEL: gpu.func @main_kernel{{.*}}memref<4x8x512x64xbf16>
# CHECK-NOT: gpu.alloc
# CHECK-SAME: known_block_size = array<i32: 64, 1, 1>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.load_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK: xegpu.store_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK-NEXT: xegpu.store_nd {{.*}}layout = #xegpu.layout<sg_layout = [4, 1], sg_data = [32, 32]>
# CHECK-NOT: gpu.alloc

import torch
import torch.nn as nn

from llama_rope import get_rope_tables, rotate_half


batch_size = 4
n_ctx = 512
dim = 2048
num_kv_heads = 8
d_head = 64


class Model(nn.Module):
    """Llama 3.2 1B K projection followed by a separate RoPE kernel.

    The input x is already normalized: [batch, sequence, dim]. Precomputed
    cos/sin tables [sequence, d_head] are shared across the batch. Returns
    rotated K [batch, num_kv_heads, sequence, d_head]. Q, V, RMSNorm, and
    attention are excluded to keep a single result tensor.
    """

    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(dim, num_kv_heads * d_head, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = x.shape
        k = self.k_proj(x).view(batch, sequence, num_kv_heads, d_head).transpose(1, 2)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return k * cos + rotate_half(k) * sin


def get_inputs():
    x = torch.randn(batch_size, n_ctx, dim)
    cos, sin = get_rope_tables(torch.arange(n_ctx))
    return [x, cos, sin]


def get_init_inputs():
    return []

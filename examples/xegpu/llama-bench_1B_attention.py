# RUN: %PYTHON %S/kernel_bench.py --input %s --dump-kernel xegpu-wg | FileCheck %s --implicit-check-not=memref.alloc --implicit-check-not=memref.copy --implicit-check-not=gpu.alloc --implicit-check-not=math.log
# REQUIRES: torch, torch_mlir

# CHECK: gpu.func @main_kernel
# CHECK-SAME: known_grid_size = array<i32: 32, 4, 4>
# CHECK: %[[QUERY_BLOCK:[a-zA-Z0-9_]+]] = gpu.block_id z
# CHECK: %[[QUERY:[a-zA-Z0-9_]+]] = arith.muli %[[QUERY_BLOCK]],
# CHECK-NOT: scf.for
# CHECK: vector.broadcast %[[QUERY]] : index
# CHECK-NOT: scf.for
# CHECK: scf.for %[[KEY:[a-zA-Z0-9_]+]] =
# CHECK: vector.broadcast %[[KEY]] : index
# CHECK: arith.cmpi sge, {{.*}} : vector<128x64xi64>
# CHECK-NEXT: arith.select
# CHECK-NOT: math.log
# CHECK: xegpu.dpas
# CHECK: vector.multi_reduction <maxnumf>
# CHECK: math.exp
# CHECK: xegpu.dpas
# CHECK: xegpu.store_nd
# CHECK-NOT: scf.for
# CHECK-NOT: gpu.func

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Llama3.2-1B scaled dot-product attention (GQA + causal), the realistic
    counterpart to KernelBench level1-97.

    Differences from the synthetic level1-97 SDPA:

      * d_head = 64 (Llama3.2-1B), the per-head reduction dimension.
      * Grouped-query attention: 32 query heads share 8 key/value heads
        (num_heads / num_kv_heads = 4). K/V carry fewer heads than Q and are
        broadcast across the group (enable_gqa=True).
      * Explicit causal masking, as in autoregressive decoding/prefill.

    Torch-MLIR 20261001 can fail to import SDPA's internal is_causal=True mask
    on XPU: its boolean aten.empty.memory_format cannot be legalized. Build the
    same mask from indices and pass is_causal=False to avoid that allocation.
    SDPA's boolean-mask conversion also failed with a scalar f64
    torch.vtensor.literal on XPU. Supply an additive mask in Q's dtype, derived
    from the predicate rather than new_zeros (which hit empty.memory_format too).
    Selecting between 0-d zero and -inf constants imports as a single
    elementwise select, so no logarithm or other workaround op reaches the IR.

    RoPE is intentionally not applied here. Q/K/V are taken as
    already-projected inputs, matching level1-97.
    Query tiles are independent GPU blocks; only the K/V sequence is serial
    within each block.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # Q: (batch, num_heads, n_ctx, d_head); K/V: (batch, num_kv_heads, n_ctx, d_head)
        rows = torch.arange(Q.shape[-2], device=Q.device)
        cols = torch.arange(K.shape[-2], device=K.device)
        allowed = rows[:, None] >= cols[None, :]
        zero = torch.tensor(0.0, dtype=Q.dtype)
        neg_inf = torch.tensor(float("-inf"), dtype=Q.dtype)
        mask = torch.where(allowed, zero, neg_inf)
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, is_causal=False, enable_gqa=True
        )


# Llama3.2-1B attention config.
batch_size = 4
n_ctx = 512
num_heads = 32  # query heads
num_kv_heads = 8  # grouped key/value heads (GQA group size = 4)
d_head = 64  # per-head reduction dim


def get_inputs():
    Q = torch.rand(batch_size, num_heads, n_ctx, d_head)
    K = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    V = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    return [Q, K, V]


def get_init_inputs():
    return []

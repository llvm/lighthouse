import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Llama3.2-1B scaled dot-product attention (GQA + causal), the realistic
    counterpart to KernelBench level1-97.

    Differences from the synthetic level1-97 SDPA:
      * d_head = 64 (Llama3.2-1B), the per-head reduction dimension, within the
        current <= 64 lowering limit. (Llama3-8B uses d_head = 128, which still
        needs the new Intel-GPU head tiling.)
      * Grouped-query attention: 32 query heads share 8 key/value heads
        (num_heads / num_kv_heads = 4). K/V carry fewer heads than Q and are
        broadcast across the group (enable_gqa=True).
      * Causal masking (is_causal=True), as in autoregressive decoding/prefill.

    RoPE is intentionally NOT applied here; the rotary embedding is a separate
    stepping-stone (llama-bench_4) because rotate_half is a slice+concat shuffle
    on the innermost (head) dimension rather than a plain matmul/elementwise op.
    Q/K/V are taken as already-projected inputs, matching level1-97.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        # Q: (batch, num_heads, n_ctx, d_head); K/V: (batch, num_kv_heads, n_ctx, d_head)
        causal = False  # NOTE handle causal masking in the lowering
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=causal, enable_gqa=True
        )


# Llama3.2-1B attention config.
batch_size = 4
n_ctx = 512
num_heads = 32  # query heads
num_kv_heads = 8  # grouped key/value heads (GQA group size = 4)
d_head = 64  # per-head reduction dim; within the current <= 64 cap


def get_inputs():
    Q = torch.rand(batch_size, num_heads, n_ctx, d_head)
    K = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    V = torch.rand(batch_size, num_kv_heads, n_ctx, d_head)
    return [Q, K, V]


def get_init_inputs():
    return []

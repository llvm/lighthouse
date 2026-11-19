import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def get_rope_tables(
    position_ids: torch.Tensor, d_head: int = 64, scaling_factor: float = 32.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Llama 3 half-split RoPE tables for a 1D position tensor."""
    if d_head <= 0 or d_head % 2:
        raise ValueError("d_head must be a positive even number")
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")
    inv_freq = 1.0 / (
        500000.0
        ** (
            torch.arange(0, d_head, 2, dtype=torch.float32, device=position_ids.device)
            / d_head
        )
    )
    wavelength = 2 * math.pi / inv_freq
    # Llama 3 scaling: original context=8192, low/high factors=1/4.
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    interpolated = (1.0 - smooth) * inv_freq / scaling_factor + smooth * inv_freq
    inv_freq = torch.where(
        wavelength < 2048.0,
        inv_freq,
        torch.where(wavelength > 8192.0, inv_freq / scaling_factor, interpolated),
    )
    angles = position_ids.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()

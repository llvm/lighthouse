import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def get_rope_tables(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Llama 3.2 1B's half-split RoPE tables for a 1D position tensor."""
    inv_freq = 1.0 / (
        500000.0
        ** (
            torch.arange(0, 64, 2, dtype=torch.float32, device=position_ids.device) / 64
        )
    )
    wavelength = 2 * math.pi / inv_freq
    # Llama 3 scaling: factor=32, original context=8192, low/high factors=1/4.
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    interpolated = (1.0 - smooth) * inv_freq / 32.0 + smooth * inv_freq
    inv_freq = torch.where(
        wavelength < 2048.0,
        inv_freq,
        torch.where(wavelength > 8192.0, inv_freq / 32.0, interpolated),
    )
    angles = position_ids.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()

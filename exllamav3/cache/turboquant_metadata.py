"""
TurboQuant KV cache quantization for exllamav3.

Adapted from turboquant-vllm (https://github.com/varjoranta/turboquant-vllm).

Algorithm: PolarQuant (Lloyd's codebook on Gaussian N(0,1/sqrt(d)) after WHT
rotation) with optional QJL 1-bit residual for K cache.

All dimensions treated uniformly (no outlier groups). Norms stored separately.
Codebook and rotation signs are deterministic from seed.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

# ── Constants ─────────────────────────────────────────────────────────────────

QJL_SCALE = 1.2533141373155002  # sqrt(pi/2)
DEFAULT_SEED = 42
K_SEED_OFFSET = 0
V_SEED_OFFSET = 500
QJL_SEED_OFFSET = 1000


# ── Codebook (Lloyd's algorithm for Gaussian) ────────────────────────────────

def compute_codebook(bit_width: int, head_dim: int) -> tuple[list[float], list[float]]:
    """Compute optimal centroids and boundaries for N(0, 1/sqrt(d)).

    Returns (centroids, boundaries) as sorted lists.
    Matches turboquant_plus.codebook exactly.
    """
    from scipy import stats

    n = 1 << bit_width
    sigma = 1.0 / math.sqrt(head_dim)

    # Special cases (hardcoded for speed and precision)
    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * head_dim))
        return [-c, c], [0.0]
    if bit_width == 2:
        s = math.sqrt(head_dim)
        centroids = [-1.51 / s, -0.453 / s, 0.453 / s, 1.51 / s]
        return centroids, [(centroids[i] + centroids[i + 1]) / 2 for i in range(3)]

    # General: quantile-based initialization + Lloyd iterations
    boundaries = list(stats.norm.ppf([i / n for i in range(1, n)], scale=sigma))
    centroids = [0.0] * n

    def cond_exp(a: float, b: float) -> float:
        a_s = a / sigma if math.isfinite(a) else a
        b_s = b / sigma if math.isfinite(b) else b
        if not math.isfinite(a_s):
            prob = stats.norm.cdf(b_s)
        elif not math.isfinite(b_s):
            prob = stats.norm.sf(a_s)
        else:
            prob = stats.norm.cdf(b_s) - stats.norm.cdf(a_s)
        if prob < 1e-15:
            return ((a if math.isfinite(a) else 0) + (b if math.isfinite(b) else 0)) / 2
        return sigma * (stats.norm.pdf(a_s) - stats.norm.pdf(b_s)) / prob

    for _ in range(100):
        centroids[0] = cond_exp(-math.inf, boundaries[0])
        for i in range(1, n - 1):
            centroids[i] = cond_exp(boundaries[i - 1], boundaries[i])
        centroids[-1] = cond_exp(boundaries[-1], math.inf)
        boundaries = [(centroids[i] + centroids[i + 1]) / 2 for i in range(n - 1)]

    centroids.sort()
    return centroids, boundaries


# ── WHT rotation signs ───────────────────────────────────────────────────────

def generate_wht_signs(
    head_dim: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic WHT rotation sign vectors.

    Returns (signs1, signs2) as float32 tensors of ±1.
    D1·H·D2 rotation: multiply by signs1, apply WHT, multiply by signs2.
    """
    gen = torch.Generator().manual_seed(seed)
    signs1 = (torch.randint(0, 2, (head_dim,), generator=gen) * 2 - 1).float()
    signs2 = (torch.randint(0, 2, (head_dim,), generator=gen) * 2 - 1).float()
    return signs1, signs2


# ── QJL matrix ────────────────────────────────────────────────────────────────

def generate_qjl_matrix(head_dim: int, seed: int) -> torch.Tensor:
    """Generate random Gaussian projection matrix for QJL."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(head_dim, head_dim, generator=gen)


# ── Packed dimension computation ──────────────────────────────────────────────

def packed_dim(bit_width: int, head_dim: int) -> int:
    """Bytes per vector in packed cache for a given PolarQuant bit width."""
    if bit_width == 4:
        return head_dim // 2
    if bit_width == 2:
        return head_dim // 4
    # 3-bit: 1 byte per index (no sub-byte packing in CUDA kernel)
    return head_dim


# ── Python reference quantize/dequantize ──────────────────────────────────────

def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform for power-of-2 last dimension.
    Normalized by 1/sqrt(n).
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, "dimension must be power of 2"
    out = x.clone()
    h = 1
    while h < n:
        out_view = out.reshape(*out.shape[:-1], -1, 2 * h)
        left = out_view[..., :h].clone()
        right = out_view[..., h:].clone()
        out_view[..., :h] = left + right
        out_view[..., h:] = left - right
        h <<= 1
    return out * (1.0 / math.sqrt(n))


def quantize_pq(
    x: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PolarQuant encode: FP16/FP32 vectors → (indices, norms).

    x: (..., head_dim) float
    Returns: (indices uint8 (..., head_dim), norms float32 (...,))
    """
    x_f32 = x.to(torch.float32)

    # L2 norm
    norms = x_f32.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    unit = x_f32 / norms

    # Rotate: D1 · H · D2
    rotated = unit * signs1
    rotated = _fwht_pow2(rotated)
    rotated = rotated * signs2

    # Searchsorted against boundaries
    indices = torch.searchsorted(boundaries, rotated).clamp(0, centroids.shape[0] - 1)

    return indices.to(torch.uint8), norms.squeeze(-1)


def dequantize_pq(
    indices: torch.Tensor,
    norms: torch.Tensor,
    signs1: torch.Tensor,
    signs2: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """PolarQuant decode: (indices, norms) → FP16 vectors.

    indices: (..., head_dim) uint8
    norms: (...,) float32
    Returns: (..., head_dim) float16
    """
    # Centroid lookup
    values = centroids[indices.long()]

    # Inverse rotation: D2 · H · D1
    values = values * signs2
    values = _fwht_pow2(values)
    values = values * signs1

    # Scale by norm
    values = values * norms.unsqueeze(-1)
    return values.to(torch.float16)


def pack_indices(indices: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Pack codebook indices into bytes.

    Matches the CUDA kernel's packing layout exactly.
    """
    shape = indices.shape[:-1]
    flat = indices.reshape(-1, head_dim).to(torch.int32)
    pd = packed_dim(bit_width, head_dim)
    out = torch.zeros(flat.shape[0], pd, dtype=torch.uint8, device=indices.device)

    if bit_width == 4:
        # 2 indices per byte: lo nibble + hi nibble
        for i in range(head_dim // 2):
            lo = flat[:, i * 2] & 0xF
            hi = flat[:, i * 2 + 1] & 0xF
            out[:, i] = (lo | (hi << 4)).to(torch.uint8)
    elif bit_width == 2:
        # 4 indices per byte
        for i in range(head_dim // 4):
            byte_val = torch.zeros(flat.shape[0], dtype=torch.int32, device=indices.device)
            for j in range(4):
                byte_val |= (flat[:, i * 4 + j] & 0x3) << (j * 2)
            out[:, i] = byte_val.to(torch.uint8)
    else:
        # 3-bit or 1-bit: 1 byte per index
        out = flat.to(torch.uint8)

    return out.reshape(*shape, pd)


def unpack_indices(packed: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Unpack bytes to codebook indices. Inverse of pack_indices."""
    shape = packed.shape[:-1]
    pd = packed_dim(bit_width, head_dim)
    flat = packed.reshape(-1, pd).to(torch.int32)
    out = torch.zeros(flat.shape[0], head_dim, dtype=torch.int32, device=packed.device)

    if bit_width == 4:
        for i in range(head_dim // 2):
            byte_val = flat[:, i]
            out[:, i * 2] = byte_val & 0xF
            out[:, i * 2 + 1] = (byte_val >> 4) & 0xF
    elif bit_width == 2:
        for i in range(head_dim // 4):
            byte_val = flat[:, i]
            for j in range(4):
                out[:, i * 4 + j] = (byte_val >> (j * 2)) & 0x3
    else:
        out = flat

    return out.reshape(*shape, head_dim).to(torch.uint8)

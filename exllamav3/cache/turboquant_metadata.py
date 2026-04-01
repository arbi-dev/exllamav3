"""
PolarQuant KV cache quantization utilities for exllamav3.

Algorithm: Lloyd's codebook quantization on WHT-rotated unit vectors.
  encode: x -> normalize -> D1*H*D2 rotation -> searchsorted -> packed indices + norm
  decode: unpack -> centroid lookup -> D2*H*D1 inverse rotation -> scale by norm

All dimensions treated uniformly. Norms stored separately as fp32.
Codebook and rotation signs are deterministic from seed.

Adapted from turboquant-vllm (https://github.com/varjoranta/turboquant-vllm).
"""

from __future__ import annotations

import math

import torch

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED = 42
K_SEED_OFFSET = 0
V_SEED_OFFSET = 500


def _next_pow2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ── Codebook (Lloyd's algorithm for Gaussian) ────────────────────────────────

def compute_codebook(bit_width: int, head_dim: int) -> tuple[list[float], list[float]]:
    """Compute optimal centroids and boundaries for N(0, 1/sqrt(d))."""
    from scipy import stats

    n = 1 << bit_width
    sigma = 1.0 / math.sqrt(head_dim)

    if bit_width == 1:
        c = math.sqrt(2.0 / (math.pi * head_dim))
        return [-c, c], [0.0]
    if bit_width == 2:
        s = math.sqrt(head_dim)
        centroids = [-1.51 / s, -0.453 / s, 0.453 / s, 1.51 / s]
        return centroids, [(centroids[i] + centroids[i + 1]) / 2 for i in range(3)]

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

def generate_wht_signs(head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic WHT rotation signs at padded dimension."""
    pd = _next_pow2(head_dim)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    s1 = (torch.randint(0, 2, (pd,), generator=gen, device="cpu") * 2 - 1).float()
    s2 = (torch.randint(0, 2, (pd,), generator=gen, device="cpu") * 2 - 1).float()
    return s1, s2


# ── Packed dimension computation ──────────────────────────────────────────────

def padded_dim(head_dim: int) -> int:
    return _next_pow2(head_dim)


def packed_dim(bit_width: int, head_dim: int) -> int:
    """Bytes per vector in packed cache."""
    pd = padded_dim(head_dim)
    if bit_width == 4:
        return pd // 2
    if bit_width == 3:
        return (pd * 3 + 7) // 8  # proper bitstream: 3 bits per index
    if bit_width == 2:
        return pd // 4
    return pd


# ── Walsh-Hadamard Transform ─────────────────────────────────────────────────

def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Normalized Fast Walsh-Hadamard Transform. Last dim must be power of 2."""
    n = x.shape[-1]
    out = x.clone()
    h = 1
    while h < n:
        v = out.reshape(*out.shape[:-1], -1, 2 * h)
        left = v[..., :h].clone()
        right = v[..., h:].clone()
        v[..., :h] = left + right
        v[..., h:] = left - right
        h <<= 1
    return out * (1.0 / math.sqrt(n))


# ── Quantize / Dequantize ────────────────────────────────────────────────────

def quantize_pq(x, signs1, signs2, centroids, boundaries):
    """PolarQuant encode. x: (..., head_dim). Returns (indices, norms)."""
    head_dim = x.shape[-1]
    pd = padded_dim(head_dim)
    x_f32 = x.to(torch.float32)
    if pd != head_dim:
        x_f32 = torch.nn.functional.pad(x_f32, (0, pd - head_dim))
    norms = x_f32.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    unit = x_f32 / norms
    rotated = unit * signs1
    rotated = _fwht_pow2(rotated)
    rotated = rotated * signs2
    indices = torch.searchsorted(boundaries, rotated).clamp(0, centroids.shape[0] - 1)
    return indices.to(torch.uint8), norms.squeeze(-1)


def dequantize_pq(indices, norms, signs1, signs2, centroids, head_dim=None):
    """PolarQuant decode. Returns (..., head_dim) float16."""
    values = centroids[indices.long()]
    values = values * signs2
    values = _fwht_pow2(values)
    values = values * signs1
    values = values * norms.unsqueeze(-1)
    if head_dim is not None and values.shape[-1] != head_dim:
        values = values[..., :head_dim]
    return values.to(torch.float16)


# ── Bit packing ──────────────────────────────────────────────────────────────

def pack_indices(indices: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Pack codebook indices into bytes. Matches CUDA kernel layout."""
    pd = padded_dim(head_dim)
    shape = indices.shape[:-1]
    flat = indices.reshape(-1, pd).to(torch.int32)
    pd_out = packed_dim(bit_width, head_dim)
    out = torch.zeros(flat.shape[0], pd_out, dtype=torch.uint8, device=indices.device)

    if bit_width == 4:
        for i in range(pd // 2):
            lo = flat[:, i * 2] & 0xF
            hi = flat[:, i * 2 + 1] & 0xF
            out[:, i] = (lo | (hi << 4)).to(torch.uint8)
    elif bit_width == 3:
        # Proper 3-bit bitstream packing
        bit_pos = 0
        for d in range(pd):
            val = flat[:, d] & 0x7
            for b in range(3):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                out[:, byte_idx] |= (((val >> b) & 1) << bit_idx).to(torch.uint8)
                bit_pos += 1
    elif bit_width == 2:
        for i in range(pd // 4):
            byte_val = torch.zeros(flat.shape[0], dtype=torch.int32, device=indices.device)
            for j in range(4):
                byte_val |= (flat[:, i * 4 + j] & 0x3) << (j * 2)
            out[:, i] = byte_val.to(torch.uint8)

    return out.reshape(*shape, pd_out)


def unpack_indices(packed: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Unpack bytes to codebook indices. Inverse of pack_indices."""
    pd = padded_dim(head_dim)
    pd_in = packed_dim(bit_width, head_dim)
    shape = packed.shape[:-1]
    flat = packed.reshape(-1, pd_in).to(torch.int32)
    out = torch.zeros(flat.shape[0], pd, dtype=torch.int32, device=packed.device)

    if bit_width == 4:
        for i in range(pd // 2):
            byte_val = flat[:, i]
            out[:, i * 2] = byte_val & 0xF
            out[:, i * 2 + 1] = (byte_val >> 4) & 0xF
    elif bit_width == 3:
        bit_pos = 0
        for d in range(pd):
            val = torch.zeros(flat.shape[0], dtype=torch.int32, device=packed.device)
            for b in range(3):
                byte_idx = bit_pos // 8
                bit_idx = bit_pos % 8
                val |= ((flat[:, byte_idx] >> bit_idx) & 1) << b
                bit_pos += 1
            out[:, d] = val
    elif bit_width == 2:
        for i in range(pd // 4):
            byte_val = flat[:, i]
            for j in range(4):
                out[:, i * 4 + j] = (byte_val >> (j * 2)) & 0x3

    return out.reshape(*shape, pd).to(torch.uint8)

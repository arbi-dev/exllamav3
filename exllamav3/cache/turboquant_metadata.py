"""TurboKV metadata — thin shim over turbokv package.

Provides backward-compatible function signatures for the exllamav3
cache layer while delegating to turbokv for actual implementation.

Requires: pip install turbokv
"""
import math
import torch
import numpy as np

# Try importing turbokv — fall back to inline implementation if not installed
try:
    from turbokv import TurboKVCodec
    from turbokv.core import _next_pow2
    _has_turbokv = True
except ImportError:
    _has_turbokv = False

    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

DEFAULT_SEED = 42
K_SEED_OFFSET = 0
V_SEED_OFFSET = 1


def padded_dim(head_dim: int) -> int:
    return _next_pow2(head_dim)


def packed_dim(bit_width: int, head_dim: int) -> int:
    pd = padded_dim(head_dim)
    return (pd * bit_width + 7) // 8


def compute_codebook(bit_width: int, head_dim: int) -> tuple[list[float], list[float]]:
    """Return (centroids, boundaries) for the given config."""
    if _has_turbokv:
        codec = TurboKVCodec(bit_width=bit_width, head_dim=head_dim)
        return codec.k_centroids.tolist(), codec.k_boundaries.tolist()

    # Fallback: compute from scipy
    from scipy import stats
    n = 1 << bit_width
    sigma = 1.0 / math.sqrt(padded_dim(head_dim))
    dist = stats.norm(0, sigma)
    boundaries = [float(dist.ppf((i + 1) / n)) for i in range(n - 1)]
    edges = [float('-inf')] + boundaries + [float('inf')]
    centroids = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        c = float(dist.expect(lambda x: x, lb=lo, ub=hi) /
                  (dist.cdf(hi) - dist.cdf(lo)))
        centroids.append(c)
    return centroids, boundaries


def generate_wht_signs(head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (signs1, signs2) diagonal sign vectors for WHT rotation."""
    pd = padded_dim(head_dim)
    rng = np.random.RandomState(seed)
    signs1 = torch.tensor(rng.choice([-1.0, 1.0], size=pd), dtype=torch.float32)
    rng2 = np.random.RandomState(seed + 1)
    signs2 = torch.tensor(rng2.choice([-1.0, 1.0], size=pd), dtype=torch.float32)
    return signs1, signs2


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform (normalized) on last dim, must be pow2."""
    n = x.shape[-1]
    h = 1
    while h < n:
        x = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :] + x[..., 1, :]
        b = x[..., 0, :] - x[..., 1, :]
        x = torch.stack([a, b], dim=-2).view(*x.shape[:-3], n)
        h <<= 1
    return x * (n ** -0.5)


def quantize_pq(x, signs1, signs2, centroids, boundaries):
    """Encode: rotate → normalize → quantize → pack."""
    orig_shape = x.shape
    hd = x.shape[-1]
    pd = padded_dim(hd)
    flat = x.reshape(-1, hd)
    if pd > hd:
        flat = torch.nn.functional.pad(flat, (0, pd - hd))
    rotated = _fwht_pow2(flat * signs1.to(flat.device)) * signs2.to(flat.device)
    norms = rotated.norm(dim=-1, keepdim=True)
    normed = rotated / (norms + 1e-10)
    bounds_t = torch.tensor(boundaries, dtype=x.dtype, device=x.device)
    indices = torch.searchsorted(bounds_t, normed)
    return indices.view(*orig_shape[:-1], pd), norms.squeeze(-1).view(orig_shape[:-1])


def dequantize_pq(indices, norms, signs1, signs2, centroids, head_dim=None):
    """Decode: gather → inverse rotate → scale by norm."""
    pd = indices.shape[-1]
    if head_dim is None:
        head_dim = pd
    cents_t = torch.tensor(centroids, dtype=torch.float32, device=indices.device)
    vals = cents_t[indices.long()]
    unrotated = _fwht_pow2(vals * signs2.to(vals.device)) * signs1.to(vals.device)
    scaled = unrotated * norms.unsqueeze(-1)
    return scaled[..., :head_dim]


def pack_indices(indices: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Pack integer indices into byte tensor."""
    pd = padded_dim(head_dim)
    flat = indices.reshape(-1, pd).to(torch.int32)
    n = flat.shape[0]
    out_bytes = packed_dim(bit_width, head_dim)

    if bit_width == 4:
        even = flat[:, 0::2] & 0xF
        odd = (flat[:, 1::2] & 0xF) << 4
        return (even | odd).to(torch.uint8)
    elif bit_width == 2:
        result = torch.zeros(n, out_bytes, dtype=torch.uint8, device=flat.device)
        for i in range(4):
            result |= (flat[:, i::4] & 0x3) << (i * 2)
        return result
    elif bit_width == 8:
        return flat.to(torch.uint8)
    else:
        # Generic bitstream packing
        result = torch.zeros(n, out_bytes, dtype=torch.uint8, device=flat.device)
        for d in range(pd):
            val = flat[:, d] & ((1 << bit_width) - 1)
            base_bit = d * bit_width
            byte0 = base_bit // 8
            bit0 = base_bit % 8
            result[:, byte0] |= (val << bit0).to(torch.uint8)
            if bit0 + bit_width > 8:
                result[:, byte0 + 1] |= (val >> (8 - bit0)).to(torch.uint8)
        return result


def unpack_indices(packed: torch.Tensor, bit_width: int, head_dim: int) -> torch.Tensor:
    """Unpack byte tensor to integer indices."""
    pd = padded_dim(head_dim)
    flat = packed.reshape(-1, packed.shape[-1])
    n = flat.shape[0]

    if bit_width == 4:
        lo = flat & 0xF
        hi = (flat >> 4) & 0xF
        return torch.stack([lo, hi], dim=-1).reshape(n, -1)[:, :pd]
    elif bit_width == 2:
        result = torch.zeros(n, pd, dtype=torch.int32, device=flat.device)
        for i in range(4):
            result[:, i::4] = (flat >> (i * 2)) & 0x3
        return result[:, :pd]
    elif bit_width == 8:
        return flat.to(torch.int32)[:, :pd]
    else:
        mask = (1 << bit_width) - 1
        result = torch.zeros(n, pd, dtype=torch.int32, device=flat.device)
        for d in range(pd):
            base_bit = d * bit_width
            byte0 = base_bit // 8
            bit0 = base_bit % 8
            val = (flat[:, byte0].to(torch.int32) >> bit0)
            if bit0 + bit_width > 8:
                val |= flat[:, byte0 + 1].to(torch.int32) << (8 - bit0)
            result[:, d] = val & mask
        return result

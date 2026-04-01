"""
TurboQuant KV cache layer for exllamav3.

PolarQuant quantization with separate norm tensors, no outlier groups,
no calibration metadata. Deterministic from seed.

Adapted from turboquant-vllm (https://github.com/varjoranta/turboquant-vllm).
"""

from __future__ import annotations
from typing_extensions import override
import torch
import numpy as np
from ..constants import PAGE_SIZE
from ..model import Config
from .cache import CacheLayer
from .turboquant_metadata import (
    compute_codebook,
    generate_wht_signs,
    packed_dim,
    padded_dim,
    quantize_pq,
    dequantize_pq,
    pack_indices,
    unpack_indices,
    DEFAULT_SEED,
    K_SEED_OFFSET,
    V_SEED_OFFSET,
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention

# Try to load CUDA extension
_has_cuda_ext = False
try:
    from exllamav3.ext import exllamav3_ext as ext
    _has_cuda_ext = hasattr(ext, 'turboquant_encode_paged')
except ImportError:
    pass


class CacheLayer_turboquant(CacheLayer):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int = 4,
        v_bits: int = 4,
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}"
        assert 2 <= k_bits <= 4 and 2 <= v_bits <= 4, \
            "bit width must be 2, 3, or 4"

        self.k_bits = k_bits
        self.v_bits = v_bits
        self.seed = seed
        self.head_dim = attention.head_dim
        self.padded_head_dim = padded_dim(attention.head_dim)
        self.num_kv_heads = attention.num_kv_heads
        self.num_pages = max_num_tokens // PAGE_SIZE

        self.k_packed_dim = packed_dim(k_bits, self.head_dim)
        self.v_packed_dim = packed_dim(v_bits, self.head_dim)

        # Tensor shapes
        self.fp16_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim)
        self.k_cache_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.k_packed_dim)
        self.v_cache_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads, self.v_packed_dim)
        self.norms_shape = (self.num_pages, PAGE_SIZE, self.num_kv_heads)

        # Device tensors (allocated in alloc())
        self.cache_k = None
        self.cache_v = None
        self.k_norms = None
        self.v_norms = None
        self.device = None

        # Codebook and rotation (allocated in alloc())
        self.k_centroids = None
        self.k_boundaries = None
        self.v_centroids = None
        self.v_boundaries = None
        self.k_signs1 = None
        self.k_signs2 = None
        self.v_signs1 = None
        self.v_signs2 = None

    @override
    def alloc(self, device: torch.device):
        self.device = device

        # Packed index cache
        self.cache_k = torch.zeros(self.k_cache_shape, dtype=torch.uint8, device=device)
        self.cache_v = torch.zeros(self.v_cache_shape, dtype=torch.uint8, device=device)

        # Separate norm tensors (fp32)
        self.k_norms = torch.zeros(self.norms_shape, dtype=torch.float32, device=device)
        self.v_norms = torch.zeros(self.norms_shape, dtype=torch.float32, device=device)

        # Compute codebooks
        k_c, k_b = compute_codebook(self.k_bits, self.head_dim)
        v_c, v_b = compute_codebook(self.v_bits, self.head_dim)
        self.k_centroids = torch.tensor(k_c, dtype=torch.float32, device=device)
        self.k_boundaries = torch.tensor(k_b, dtype=torch.float32, device=device)
        self.v_centroids = torch.tensor(v_c, dtype=torch.float32, device=device)
        self.v_boundaries = torch.tensor(v_b, dtype=torch.float32, device=device)

        # WHT rotation signs (deterministic from seed)
        self.k_signs1, self.k_signs2 = generate_wht_signs(self.head_dim, self.seed + K_SEED_OFFSET)
        self.k_signs1 = self.k_signs1.to(device)
        self.k_signs2 = self.k_signs2.to(device)
        self.v_signs1, self.v_signs2 = generate_wht_signs(self.head_dim, self.seed + V_SEED_OFFSET)
        self.v_signs1 = self.v_signs1.to(device)
        self.v_signs2 = self.v_signs2.to(device)

    @override
    def free(self):
        self.device = None
        self.cache_k = None
        self.cache_v = None
        self.k_norms = None
        self.v_norms = None
        self.k_centroids = None
        self.k_boundaries = None
        self.v_centroids = None
        self.v_boundaries = None
        self.k_signs1 = None
        self.k_signs2 = None
        self.v_signs1 = None
        self.v_signs2 = None

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor):
        """Dequantize packed cache to FP16 for flash attention."""
        k_out = torch.zeros(self.fp16_shape, dtype=torch.half, device=self.device)
        v_out = torch.zeros(self.fp16_shape, dtype=torch.half, device=self.device)

        if _has_cuda_ext and self.device.type == "cuda":
            ext.turboquant_decode_paged(
                self.cache_k, self.cache_v,
                self.k_norms, self.v_norms,
                k_out, v_out,
                cache_seqlens.to(torch.int32),
                block_table.to(torch.int32),
                self.k_bits, self.v_bits,
                self.k_signs1, self.k_signs2,
                self.v_signs1, self.v_signs2,
                self.k_centroids, self.v_centroids,
            )
        else:
            self._decode_python(k_out, v_out, cache_seqlens, block_table)

        return k_out, v_out

    def _decode_python(self, k_out, v_out, cache_seqlens, block_table):
        batch_size = cache_seqlens.shape[0]
        for b in range(batch_size):
            seq_len = cache_seqlens[b].item()
            if seq_len == 0:
                continue
            pages_needed = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
            for p in range(pages_needed):
                mapped_page = block_table[b, p].item()
                tokens_in_page = min(PAGE_SIZE, seq_len - p * PAGE_SIZE)
                T, H = tokens_in_page, self.num_kv_heads

                k_packed = self.cache_k[mapped_page, :T]
                k_indices = unpack_indices(k_packed.reshape(T * H, self.k_packed_dim), self.k_bits, self.head_dim)
                k_fp16 = dequantize_pq(k_indices.reshape(T * H, self.head_dim), self.k_norms[mapped_page, :T].reshape(T * H), self.k_signs1, self.k_signs2, self.k_centroids)
                k_out[mapped_page, :T] = k_fp16.reshape(T, H, self.head_dim)

                v_packed = self.cache_v[mapped_page, :T]
                v_indices = unpack_indices(v_packed.reshape(T * H, self.v_packed_dim), self.v_bits, self.head_dim)
                v_fp16 = dequantize_pq(v_indices.reshape(T * H, self.head_dim), self.v_norms[mapped_page, :T].reshape(T * H), self.v_signs1, self.v_signs2, self.v_centroids)
                v_out[mapped_page, :T] = v_fp16.reshape(T, H, self.head_dim)

    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ):
        """Quantize FP16 K/V and store packed indices + norms."""
        if _has_cuda_ext and self.device.type == "cuda":
            ext.turboquant_encode_paged(
                k, v,
                self.cache_k, self.cache_v,
                self.k_norms, self.v_norms,
                cache_seqlens.to(torch.int32),
                block_table.to(torch.int32),
                length,
                self.k_bits, self.v_bits,
                self.k_signs1, self.k_signs2,
                self.v_signs1, self.v_signs2,
                self.k_boundaries, self.v_boundaries,
            )
        else:
            self._encode_python(cache_seqlens, block_table, k, v, length)

    def _encode_python(self, cache_seqlens, block_table, k, v, length):
        batch_size = cache_seqlens.shape[0]
        for b in range(batch_size):
            seq_start = cache_seqlens[b].item()
            for t_off in range(length):
                t = seq_start + t_off
                page_idx = t // PAGE_SIZE
                token_in_page = t % PAGE_SIZE
                mapped_page = block_table[b, page_idx].item()

                k_vec = k[mapped_page, token_in_page]
                k_indices, k_n = quantize_pq(k_vec, self.k_signs1, self.k_signs2, self.k_centroids, self.k_boundaries)
                self.cache_k[mapped_page, token_in_page] = pack_indices(k_indices, self.k_bits, self.head_dim)
                self.k_norms[mapped_page, token_in_page] = k_n

                v_vec = v[mapped_page, token_in_page]
                v_indices, v_n = quantize_pq(v_vec, self.v_signs1, self.v_signs2, self.v_centroids, self.v_boundaries)
                self.cache_v[mapped_page, token_in_page] = pack_indices(v_indices, self.v_bits, self.head_dim)
                self.v_norms[mapped_page, token_in_page] = v_n

    @override
    def copy_page(self, source: CacheLayer_turboquant, from_page: int, to_page: int, num_tokens: int):
        self.cache_k[to_page, :num_tokens].copy_(source.cache_k[from_page, :num_tokens], non_blocking=True)
        self.cache_v[to_page, :num_tokens].copy_(source.cache_v[from_page, :num_tokens], non_blocking=True)
        self.k_norms[to_page, :num_tokens].copy_(source.k_norms[from_page, :num_tokens], non_blocking=True)
        self.v_norms[to_page, :num_tokens].copy_(source.v_norms[from_page, :num_tokens], non_blocking=True)

    @override
    def get_tensors(self):
        return [self.cache_k, self.cache_v, self.k_norms, self.v_norms]

    @override
    def storage_size(self):
        return (
            np.prod(self.k_cache_shape)
            + np.prod(self.v_cache_shape)
            + 2 * np.prod(self.norms_shape) * 4  # fp32 norms
        )

    @override
    def overhead_size(self):
        return 2 * np.prod(self.fp16_shape) * torch.half.itemsize

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_turboquant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "seed": self.seed,
            }
        }

    @override
    def get_kv_alloc_placeholder(self):
        k = torch.empty(self.fp16_shape, dtype=torch.half, device=self.device)
        v = torch.empty(self.fp16_shape, dtype=torch.half, device=self.device)
        return k, v

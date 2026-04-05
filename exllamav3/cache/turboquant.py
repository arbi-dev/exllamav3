"""
TurboKV cache layer for exllamav3.

Architecture aligned with turbokv (https://github.com/arbi-dev/turbokv):
  - KV stored in ROTATED space (WHT applied at compress time)
  - Decode = raw dequant only (unpack + centroid * norm, no inverse WHT)
  - Q pre-rotated before attention, output post-rotated after
  - Eliminates WHT from the decode kernel (8 fewer __syncthreads)

Uses turbokv.TurboKVCodec for rotation matrices and codebooks.
Falls back to turboquant_metadata.py if turbokv not installed.
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

# Try turbokv for rotation matrices (optimized path)
try:
    from turbokv import TurboKVCodec
    _has_turbokv = True
except ImportError:
    _has_turbokv = False

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
        assert 2 <= k_bits <= 8, "bit width must be 2-8"

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

        # Codebook, rotation, and codec
        self.k_centroids = None
        self.v_centroids = None
        self.k_boundaries = None
        self.v_boundaries = None
        self.k_signs1 = None
        self.k_signs2 = None
        self.v_signs1 = None
        self.v_signs2 = None

        # TurboKV codec for pre-rotation (if available)
        self._codec = None
        # Pre-computed scaled rotation matrix: scale * R_fwd
        self._scaled_k_R_fwd = None
        self._v_R_inv = None

        # Calibration state
        self._calibrator = None
        self._calibration_mode = False
        self._calibration_tokens_seen = 0

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
        k_c, k_b = compute_codebook(self.k_bits, self.padded_head_dim)
        v_c, v_b = compute_codebook(self.v_bits, self.padded_head_dim)
        self.k_centroids = torch.tensor(k_c, dtype=torch.float32, device=device)
        self.k_boundaries = torch.tensor(k_b, dtype=torch.float32, device=device)
        self.v_centroids = torch.tensor(v_c, dtype=torch.float32, device=device)
        self.v_boundaries = torch.tensor(v_b, dtype=torch.float32, device=device)

        # WHT rotation signs (for legacy CUDA kernel path)
        self.k_signs1, self.k_signs2 = generate_wht_signs(self.head_dim, self.seed + K_SEED_OFFSET)
        self.k_signs1 = self.k_signs1.to(device)
        self.k_signs2 = self.k_signs2.to(device)
        self.v_signs1, self.v_signs2 = generate_wht_signs(self.head_dim, self.seed + V_SEED_OFFSET)
        self.v_signs1 = self.v_signs1.to(device)
        self.v_signs2 = self.v_signs2.to(device)

        # TurboKV codec — provides rotation matrices for pre-rotate Q approach
        if _has_turbokv:
            self._codec = TurboKVCodec(
                bit_width=self.k_bits,
                head_dim=self.head_dim,
                device=device,
            )
            # Pre-compute scaled rotation for Q: includes softmax scale
            sm_scale = self.attention.sm_scale if hasattr(self.attention, 'sm_scale') else (self.head_dim ** -0.5)
            self._scaled_k_R_fwd = (sm_scale * self._codec.k_R_fwd).contiguous()
            self._v_R_inv = self._codec.v_R_inv.contiguous()

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
        self._codec = None
        self._scaled_k_R_fwd = None
        self._v_R_inv = None

    # ── Calibration ───────────────────────────────────────────────────────

    def enable_calibration(self, warmup_tokens: int = 10000):
        """Enable online calibration mode.

        During calibration, the cache collects KV statistics from actual
        inference. After `warmup_tokens` tokens, call `finish_calibration()`
        to optimize centroids using Lloyd's algorithm on the real distribution.

        This improves cosine similarity from ~0.989 (Gaussian codebook) to
        ~0.998+ (data-optimized codebook).
        """
        if not _has_turbokv:
            raise RuntimeError("turbokv required for calibration")
        from turbokv.calibrate_online import OnlineCalibrator
        self._calibrator = OnlineCalibrator(
            num_layers=1,  # one calibrator per cache layer
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bit_width=self.k_bits,
        )
        self._calibration_mode = True
        self._calibration_tokens_seen = 0
        self._calibration_warmup = warmup_tokens

    def _collect_calibration(self, k: torch.Tensor, v: torch.Tensor):
        """Collect KV samples during calibration warmup."""
        if not self._calibration_mode or self._calibrator is None:
            return
        self._calibrator.collect(
            layer_idx=0,
            key=k,
            value=v,
            R_fwd_k=self._codec.k_R_fwd,
            R_fwd_v=self._codec.v_R_fwd,
        )
        self._calibration_tokens_seen += k.shape[0]

    def finish_calibration(self):
        """Optimize centroids from collected data and apply.

        Call after warmup inference. The calibrated codebook replaces the
        default Gaussian codebook for all subsequent encode/decode.
        """
        if self._calibrator is None or not self._calibrator.has_enough(0):
            return False

        k_cent, k_bound, v_cent, v_bound = self._calibrator.optimize(
            layer_idx=0, device=self.device
        )
        self.k_centroids = k_cent
        self.k_boundaries = k_bound
        self.v_centroids = v_cent
        self.v_boundaries = v_bound

        # Update codec if available
        if self._codec is not None:
            self._codec.k_centroids = k_cent
            self._codec.k_boundaries = k_bound
            self._codec.v_centroids = v_cent
            self._codec.v_boundaries = v_bound

        self._calibration_mode = False
        return True

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_mode

    @property
    def calibration_ready(self) -> bool:
        return (self._calibrator is not None
                and self._calibrator.has_enough(0))

    # ── Pre-rotation hooks (called by attention module) ──────────────────

    def pre_rotate_q(self, q: torch.Tensor) -> torch.Tensor:
        """Pre-rotate Q by scaled k_R_fwd. Call before flash_attn."""
        if self._scaled_k_R_fwd is not None:
            # q: (bsz, seqlen, num_q_heads, head_dim)
            return torch.matmul(
                q.to(torch.float32), self._scaled_k_R_fwd
            ).to(q.dtype)
        return q

    def post_rotate_output(self, o: torch.Tensor) -> torch.Tensor:
        """Post-rotate output by v_R_inv. Call after flash_attn."""
        if self._v_R_inv is not None:
            return torch.matmul(
                o.to(torch.float32), self._v_R_inv
            ).to(o.dtype)
        return o

    @property
    def has_pre_rotation(self) -> bool:
        """Whether this cache layer uses the pre-rotation trick."""
        return self._scaled_k_R_fwd is not None

    # ── Cache operations ─────────────────────────────────────────────────

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor):
        """Dequantize packed cache to FP16 for flash attention.

        If turbokv is available and pre-rotation is active, returns KV
        in rotated space (no WHT inverse needed — Q is pre-rotated).
        Otherwise falls back to full dequant with WHT inverse.
        """
        k_out = torch.zeros(self.fp16_shape, dtype=torch.half, device=self.device)
        v_out = torch.zeros(self.fp16_shape, dtype=torch.half, device=self.device)

        if self.has_pre_rotation:
            # Fast path: raw dequant (no WHT). KV stays in rotated space.
            self._decode_raw(k_out, v_out, cache_seqlens, block_table)
        elif _has_cuda_ext and self.device.type == "cuda":
            # Legacy CUDA path: full dequant with WHT inverse
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
            # Python fallback with full WHT
            self._decode_python(k_out, v_out, cache_seqlens, block_table)

        return k_out, v_out

    def _decode_raw(self, k_out, v_out, cache_seqlens, block_table):
        """Raw dequant: unpack + centroid * norm. No WHT inverse.

        Used when pre-rotation is active (Q pre-rotated, KV stored rotated).
        """
        bsz = cache_seqlens.shape[0]
        for b in range(bsz):
            sl = cache_seqlens[b].item()
            if sl == 0:
                continue
            pages_needed = (sl + PAGE_SIZE - 1) // PAGE_SIZE
            for p in range(pages_needed):
                page_idx = block_table[b, p].item()
                tokens_in_page = min(PAGE_SIZE, sl - p * PAGE_SIZE)
                for t in range(tokens_in_page):
                    for h in range(self.num_kv_heads):
                        # K: unpack + centroid lookup + norm multiply
                        k_packed = self.cache_k[page_idx, t, h]
                        k_norm = self.k_norms[page_idx, t, h].item()
                        k_indices = unpack_indices(
                            k_packed.unsqueeze(0), self.k_bits, self.head_dim
                        ).squeeze(0)
                        k_vals = self.k_centroids[k_indices.long()] * k_norm
                        k_out[page_idx, t, h, :self.head_dim] = k_vals[:self.head_dim].half()

                        # V: same
                        v_packed = self.cache_v[page_idx, t, h]
                        v_norm = self.v_norms[page_idx, t, h].item()
                        v_indices = unpack_indices(
                            v_packed.unsqueeze(0), self.v_bits, self.head_dim
                        ).squeeze(0)
                        v_vals = self.v_centroids[v_indices.long()] * v_norm
                        v_out[page_idx, t, h, :self.head_dim] = v_vals[:self.head_dim].half()

    def _decode_python(self, k_out, v_out, cache_seqlens, block_table):
        """Full dequant with WHT inverse (legacy Python fallback)."""
        bsz = cache_seqlens.shape[0]
        for b in range(bsz):
            sl = cache_seqlens[b].item()
            if sl == 0:
                continue
            pages_needed = (sl + PAGE_SIZE - 1) // PAGE_SIZE
            for p in range(pages_needed):
                page_idx = block_table[b, p].item()
                tokens_in_page = min(PAGE_SIZE, sl - p * PAGE_SIZE)
                for t in range(tokens_in_page):
                    for h in range(self.num_kv_heads):
                        k_packed = self.cache_k[page_idx, t, h]
                        k_norm = self.k_norms[page_idx, t, h]
                        k_idx = unpack_indices(
                            k_packed.unsqueeze(0), self.k_bits, self.head_dim
                        ).squeeze(0)
                        k_recon = dequantize_pq(
                            k_idx, k_norm.unsqueeze(0),
                            self.k_signs1, self.k_signs2,
                            self.k_centroids.tolist(), self.head_dim,
                        )
                        k_out[page_idx, t, h, :self.head_dim] = k_recon.squeeze(0)[:self.head_dim].half()

                        v_packed = self.cache_v[page_idx, t, h]
                        v_norm = self.v_norms[page_idx, t, h]
                        v_idx = unpack_indices(
                            v_packed.unsqueeze(0), self.v_bits, self.head_dim
                        ).squeeze(0)
                        v_recon = dequantize_pq(
                            v_idx, v_norm.unsqueeze(0),
                            self.v_signs1, self.v_signs2,
                            self.v_centroids.tolist(), self.head_dim,
                        )
                        v_out[page_idx, t, h, :self.head_dim] = v_recon.squeeze(0)[:self.head_dim].half()

    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        length: int,
    ):
        """Compress and store new KV tokens.

        If turbokv codec is available, uses it for rotation + quantization.
        Otherwise falls back to legacy encode path.
        """
        k = cache_k[:, :length, :, :self.head_dim].contiguous()
        v = cache_v[:, :length, :, :self.head_dim].contiguous()

        # Collect calibration data during warmup
        if self._calibration_mode:
            self._collect_calibration(
                k.reshape(-1, self.num_kv_heads, self.head_dim),
                v.reshape(-1, self.num_kv_heads, self.head_dim),
            )

        if self._codec is not None and self._codec.device == self.device:
            # Turbokv path: codec handles rotation + quantize + pack
            self._encode_turbokv(cache_seqlens, block_table, k, v, length)
        elif _has_cuda_ext and self.device.type == "cuda":
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

    def _encode_turbokv(self, cache_seqlens, block_table, k, v, length):
        """Encode using turbokv codec (rotation + quantize + pack)."""
        bsz = cache_seqlens.shape[0]
        for b in range(bsz):
            seq_start = cache_seqlens[b].item()
            for t in range(length):
                token_pos = seq_start + t
                page_idx = block_table[b, token_pos // PAGE_SIZE].item()
                entry_idx = token_pos % PAGE_SIZE

                # K: compress via codec (handles rotation internally)
                k_vec = k[b, t].unsqueeze(0)  # (1, num_kv_heads, head_dim)
                k_packed, k_norms = self._codec.compress_k(k_vec)
                self.cache_k[page_idx, entry_idx] = k_packed.squeeze(0)
                self.k_norms[page_idx, entry_idx] = k_norms.squeeze(0).float()

                # V
                v_vec = v[b, t].unsqueeze(0)
                v_packed, v_norms = self._codec.compress_v(v_vec)
                self.cache_v[page_idx, entry_idx] = v_packed.squeeze(0)
                self.v_norms[page_idx, entry_idx] = v_norms.squeeze(0).float()

    def _encode_python(self, cache_seqlens, block_table, k, v, length):
        """Legacy Python encode with WHT rotation."""
        bsz = cache_seqlens.shape[0]
        for b in range(bsz):
            seq_start = cache_seqlens[b].item()
            for t in range(length):
                token_pos = seq_start + t
                page_idx = block_table[b, token_pos // PAGE_SIZE].item()
                entry_idx = token_pos % PAGE_SIZE

                for h in range(self.num_kv_heads):
                    # K
                    k_vec = k[b, t, h]
                    k_idx, k_norm = quantize_pq(
                        k_vec.unsqueeze(0).float(),
                        self.k_signs1, self.k_signs2,
                        self.k_centroids.tolist(), self.k_boundaries.tolist(),
                    )
                    k_packed = pack_indices(k_idx, self.k_bits, self.head_dim)
                    self.cache_k[page_idx, entry_idx, h] = k_packed.squeeze(0)
                    self.k_norms[page_idx, entry_idx, h] = k_norm.squeeze(0)

                    # V
                    v_vec = v[b, t, h]
                    v_idx, v_norm = quantize_pq(
                        v_vec.unsqueeze(0).float(),
                        self.v_signs1, self.v_signs2,
                        self.v_centroids.tolist(), self.v_boundaries.tolist(),
                    )
                    v_packed = pack_indices(v_idx, self.v_bits, self.head_dim)
                    self.cache_v[page_idx, entry_idx, h] = v_packed.squeeze(0)
                    self.v_norms[page_idx, entry_idx, h] = v_norm.squeeze(0)

    @override
    def copy_page(self, source: CacheLayer_turboquant, from_page: int, to_page: int, num_tokens: int):
        self.cache_k[to_page, :num_tokens] = source.cache_k[from_page, :num_tokens]
        self.cache_v[to_page, :num_tokens] = source.cache_v[from_page, :num_tokens]
        self.k_norms[to_page, :num_tokens] = source.k_norms[from_page, :num_tokens]
        self.v_norms[to_page, :num_tokens] = source.v_norms[from_page, :num_tokens]

    def get_tensors(self):
        return [self.cache_k, self.cache_v, self.k_norms, self.v_norms]

    def storage_size(self):
        k_bytes = self.num_pages * PAGE_SIZE * self.num_kv_heads * self.k_packed_dim
        v_bytes = self.num_pages * PAGE_SIZE * self.num_kv_heads * self.v_packed_dim
        n_bytes = 2 * self.num_pages * PAGE_SIZE * self.num_kv_heads * 4  # fp32 norms
        return k_bytes + v_bytes + n_bytes

    def overhead_size(self):
        """Non-cache overhead: codebook + rotation + decompress buffers."""
        return 0  # Negligible — a few KB

    def tp_export(self, plan):
        return {
            "type": "turboquant",
            "cache_id": self.cache_id,
            "k_bits": self.k_bits,
            "v_bits": self.v_bits,
            "seed": self.seed,
        }

    def get_kv_alloc_placeholder(self):
        k = torch.empty(self.fp16_shape, dtype=torch.half, device=self.device)
        v = torch.empty(self.fp16_shape, dtype=torch.half, device=self.device)
        return k, v

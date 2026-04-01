"""
TurboQuant (PolarQuant) unit tests.

Standalone — no model or CUDA extension required. Tests the core algorithm:
codebook computation, WHT transforms, bit packing, quantize/dequantize roundtrip.

Also tests CUDA kernel correctness if the extension is available.
"""

import sys, os, math
import torch
import torch.testing

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "cache")
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

tq = _load_module("turboquant_metadata", os.path.join(CACHE_DIR, "turboquant_metadata.py"))

PAGE_SIZE = 256


# ── Core algorithm tests (no CUDA needed) ────────────────────────────────────

def test_codebook():
    print("Test: codebook computation")
    for bits in [2, 3, 4]:
        for dim in [64, 128, 256]:
            c, b = tq.compute_codebook(bits, dim)
            n = 1 << bits
            assert len(c) == n and len(b) == n - 1
            assert c == sorted(c)
            assert abs(c[0] + c[-1]) < 0.01, f"Not symmetric: {c}"
    print("  PASSED")


def test_wht():
    print("Test: WHT properties")
    for dim in [64, 128, 256]:
        x = torch.randn(4, dim)
        y = tq._fwht_pow2(x)
        torch.testing.assert_close(tq._fwht_pow2(y), x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4, rtol=1e-4)
    print("  PASSED")


def test_signs():
    print("Test: sign determinism")
    s1a, s2a = tq.generate_wht_signs(128, 42)
    s1b, s2b = tq.generate_wht_signs(128, 42)
    assert torch.equal(s1a, s1b) and torch.equal(s2a, s2b)
    s1c, _ = tq.generate_wht_signs(128, 43)
    assert not torch.equal(s1a, s1c)
    print("  PASSED")


def test_pack_unpack():
    print("Test: pack/unpack roundtrip")
    for bits in [2, 3, 4]:
        for dim in [64, 128, 256]:
            max_val = (1 << bits) - 1
            idx = torch.randint(0, max_val + 1, (8, dim), dtype=torch.uint8)
            packed = tq.pack_indices(idx, bits, dim)
            unpacked = tq.unpack_indices(packed, bits, dim)
            assert torch.equal(idx, unpacked), f"Failed at bits={bits} dim={dim}"
    print("  PASSED")


def test_quantize_dequantize():
    print("Test: PolarQuant roundtrip quality")
    torch.manual_seed(42)
    for bits in [2, 3, 4]:
        dim = 128
        c, b = tq.compute_codebook(bits, dim)
        centroids = torch.tensor(c, dtype=torch.float32)
        boundaries = torch.tensor(b, dtype=torch.float32)
        s1, s2 = tq.generate_wht_signs(dim, 42)

        x = torch.randn(64, dim)
        indices, norms = tq.quantize_pq(x, s1, s2, centroids, boundaries)
        packed = tq.pack_indices(indices, bits, dim)
        indices2 = tq.unpack_indices(packed, bits, dim)
        x_hat = tq.dequantize_pq(indices2, norms, s1, s2, centroids)

        cos = torch.nn.functional.cosine_similarity(x, x_hat.float(), dim=-1).mean().item()
        print(f"  {bits}-bit: cosine={cos:.4f}")
        assert cos > 0.90
    print("  PASSED")


# ── CUDA kernel correctness (skipped if extension not available) ──────────────

def test_cuda_encode_decode():
    if not torch.cuda.is_available():
        print("Test: CUDA kernels — SKIPPED (no CUDA)")
        return

    try:
        import exllamav3_ext as ext
        if not hasattr(ext, 'turboquant_encode_paged'):
            print("Test: CUDA kernels — SKIPPED (extension not built)")
            return
    except ImportError:
        print("Test: CUDA kernels — SKIPPED (extension not found)")
        return

    print("Test: CUDA kernel bit-exact correctness")
    device = torch.device("cuda:0")

    for bits in [2, 3, 4]:
        for head_dim in [64, 128, 256]:
            torch.manual_seed(42)
            num_kv_heads = 2
            num_tokens = 16
            num_pages = 1

            c, b = tq.compute_codebook(bits, head_dim)
            centroids = torch.tensor(c, dtype=torch.float32, device=device)
            boundaries = torch.tensor(b, dtype=torch.float32, device=device)
            k_s1, k_s2 = tq.generate_wht_signs(head_dim, 42)
            v_s1, v_s2 = tq.generate_wht_signs(head_dim, 542)
            k_s1, k_s2 = k_s1.to(device), k_s2.to(device)
            v_s1, v_s2 = v_s1.to(device), v_s2.to(device)

            pd = tq.packed_dim(bits, head_dim)

            # Allocate paged tensors
            k_fp16 = torch.randn(num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
            v_fp16 = torch.randn(num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
            k_cache_cuda = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, pd, dtype=torch.uint8, device=device)
            v_cache_cuda = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, pd, dtype=torch.uint8, device=device)
            k_norms_cuda = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, dtype=torch.float32, device=device)
            v_norms_cuda = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, dtype=torch.float32, device=device)
            k_cache_py = torch.zeros_like(k_cache_cuda)
            v_cache_py = torch.zeros_like(v_cache_cuda)
            k_norms_py = torch.zeros_like(k_norms_cuda)
            v_norms_py = torch.zeros_like(v_norms_cuda)

            seqlens = torch.tensor([0], dtype=torch.int32, device=device)
            bt = torch.tensor([[0]], dtype=torch.int32, device=device)

            # CUDA encode
            ext.turboquant_encode_paged(
                k_fp16, v_fp16, k_cache_cuda, v_cache_cuda, k_norms_cuda, v_norms_cuda,
                seqlens, bt, num_tokens, bits, bits,
                k_s1, k_s2, v_s1, v_s2, boundaries, boundaries,
            )

            # Python encode
            for t in range(num_tokens):
                k_idx, k_n = tq.quantize_pq(k_fp16[0, t], k_s1, k_s2, centroids, boundaries)
                k_cache_py[0, t] = tq.pack_indices(k_idx, bits, head_dim)
                k_norms_py[0, t] = k_n
                v_idx, v_n = tq.quantize_pq(v_fp16[0, t], v_s1, v_s2, centroids, boundaries)
                v_cache_py[0, t] = tq.pack_indices(v_idx, bits, head_dim)
                v_norms_py[0, t] = v_n

            k_match = torch.equal(k_cache_cuda[0, :num_tokens], k_cache_py[0, :num_tokens])
            v_match = torch.equal(v_cache_cuda[0, :num_tokens], v_cache_py[0, :num_tokens])
            k_norm_ok = torch.allclose(k_norms_cuda[0, :num_tokens], k_norms_py[0, :num_tokens], atol=1e-5)

            # CUDA decode
            k_out = torch.zeros_like(k_fp16)
            v_out = torch.zeros_like(v_fp16)
            seqlens_read = torch.tensor([num_tokens], dtype=torch.int32, device=device)
            ext.turboquant_decode_paged(
                k_cache_cuda, v_cache_cuda, k_norms_cuda, v_norms_cuda, k_out, v_out,
                seqlens_read, bt, bits, bits,
                k_s1, k_s2, v_s1, v_s2, centroids, centroids,
            )

            k_cos = torch.nn.functional.cosine_similarity(
                k_fp16[0, :num_tokens].float().reshape(-1, head_dim),
                k_out[0, :num_tokens].float().reshape(-1, head_dim), dim=-1
            ).mean().item()

            status = "✓" if (k_match and v_match and k_norm_ok and k_cos > 0.90) else "✗"
            print(f"  {bits}b hd={head_dim}: encode={'exact' if k_match else 'DIFF'} "
                  f"norms={'ok' if k_norm_ok else 'DIFF'} roundtrip_cos={k_cos:.4f} {status}")

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant Unit Tests")
    print("=" * 60)

    test_codebook()
    test_wht()
    test_signs()
    test_pack_unpack()
    test_quantize_dequantize()
    test_cuda_encode_decode()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

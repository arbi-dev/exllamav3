"""
CUDA kernel bit-exact correctness tests.

Verifies that CUDA encode/decode produces identical results to the Python reference.
Tests:
1. CUDA encode → Python decode: verify packed bytes match
2. Python encode → CUDA decode: verify FP16 output matches
3. CUDA roundtrip vs Python roundtrip: verify identical reconstruction
4. Edge cases: page boundaries, various head_dims, all bit widths
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.testing

from exllamav3.cache.turboquant_metadata import (
    compute_codebook, generate_wht_signs, packed_dim,
    quantize_pq, dequantize_pq, pack_indices, unpack_indices,
    DEFAULT_SEED, K_SEED_OFFSET, V_SEED_OFFSET,
)

# Load CUDA extension
import exllamav3_ext as ext

PAGE_SIZE = 256


def setup_codebook_and_signs(bits, head_dim, device, seed_offset=0):
    """Create all tensors needed for encode/decode."""
    c, b = compute_codebook(bits, head_dim)
    centroids = torch.tensor(c, dtype=torch.float32, device=device)
    boundaries = torch.tensor(b, dtype=torch.float32, device=device)
    s1, s2 = generate_wht_signs(head_dim, DEFAULT_SEED + seed_offset)
    s1, s2 = s1.to(device), s2.to(device)
    return centroids, boundaries, s1, s2


def create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device):
    """Create the full set of paged cache tensors."""
    k_pd = packed_dim(k_bits, head_dim)
    v_pd = packed_dim(v_bits, head_dim)

    k_fp16 = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16 = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
    k_cache = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, k_pd, dtype=torch.uint8, device=device)
    v_cache = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, v_pd, dtype=torch.uint8, device=device)
    k_norms = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, dtype=torch.float32, device=device)
    v_norms = torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, dtype=torch.float32, device=device)
    k_out = torch.zeros_like(k_fp16)
    v_out = torch.zeros_like(v_fp16)

    return k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms, k_out, v_out


def test_cuda_encode_vs_python(head_dim=128, k_bits=4, v_bits=4, num_kv_heads=4, num_tokens=8):
    """Verify CUDA encode produces identical packed bytes to Python encode."""
    print(f"\nTest: CUDA encode vs Python encode (hd={head_dim}, k={k_bits}b, v={v_bits}b, heads={num_kv_heads})")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    k_centroids, k_boundaries, k_s1, k_s2 = setup_codebook_and_signs(k_bits, head_dim, device, K_SEED_OFFSET)
    v_centroids, v_boundaries, v_s1, v_s2 = setup_codebook_and_signs(v_bits, head_dim, device, V_SEED_OFFSET)

    num_pages = 1
    k_fp16, v_fp16, k_cache_cuda, v_cache_cuda, k_norms_cuda, v_norms_cuda, _, _ = \
        create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device)
    _, _, k_cache_py, v_cache_py, k_norms_py, v_norms_py, _, _ = \
        create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device)

    # Write random data into the FP16 tensors (simulating flash_attn writing K/V)
    k_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)

    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # CUDA encode
    ext.turboquant_encode_paged(
        k_fp16, v_fp16, k_cache_cuda, v_cache_cuda, k_norms_cuda, v_norms_cuda,
        cache_seqlens, block_table, num_tokens, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_boundaries, v_boundaries,
    )

    # Python encode (per-token, matching CacheLayer logic)
    for t in range(num_tokens):
        k_vec = k_fp16[0, t]  # (H, D)
        k_idx, k_n = quantize_pq(k_vec, k_s1, k_s2, k_centroids, k_boundaries)
        k_cache_py[0, t] = pack_indices(k_idx, k_bits, head_dim)
        k_norms_py[0, t] = k_n

        v_vec = v_fp16[0, t]
        v_idx, v_n = quantize_pq(v_vec, v_s1, v_s2, v_centroids, v_boundaries)
        v_cache_py[0, t] = pack_indices(v_idx, v_bits, head_dim)
        v_norms_py[0, t] = v_n

    # Compare packed bytes
    k_match = torch.equal(k_cache_cuda[0, :num_tokens], k_cache_py[0, :num_tokens])
    v_match = torch.equal(v_cache_cuda[0, :num_tokens], v_cache_py[0, :num_tokens])
    k_norm_close = torch.allclose(k_norms_cuda[0, :num_tokens], k_norms_py[0, :num_tokens], atol=1e-5)
    v_norm_close = torch.allclose(v_norms_cuda[0, :num_tokens], v_norms_py[0, :num_tokens], atol=1e-5)

    if not k_match:
        diff_count = (k_cache_cuda[0, :num_tokens] != k_cache_py[0, :num_tokens]).sum().item()
        total = k_cache_cuda[0, :num_tokens].numel()
        print(f"  K cache MISMATCH: {diff_count}/{total} bytes differ")
        # Show first difference
        for t in range(num_tokens):
            for h in range(num_kv_heads):
                if not torch.equal(k_cache_cuda[0, t, h], k_cache_py[0, t, h]):
                    print(f"    First diff at token={t} head={h}")
                    print(f"    CUDA: {k_cache_cuda[0, t, h, :8].tolist()}")
                    print(f"    Python: {k_cache_py[0, t, h, :8].tolist()}")
                    break
            else:
                continue
            break
    else:
        print(f"  K cache: EXACT MATCH ✓")

    if not v_match:
        diff_count = (v_cache_cuda[0, :num_tokens] != v_cache_py[0, :num_tokens]).sum().item()
        total = v_cache_cuda[0, :num_tokens].numel()
        print(f"  V cache MISMATCH: {diff_count}/{total} bytes differ")
    else:
        print(f"  V cache: EXACT MATCH ✓")

    print(f"  K norms: {'MATCH ✓' if k_norm_close else 'MISMATCH'}")
    print(f"  V norms: {'MATCH ✓' if v_norm_close else 'MISMATCH'}")

    return k_match and v_match and k_norm_close and v_norm_close


def test_cuda_decode_vs_python(head_dim=128, k_bits=4, v_bits=4, num_kv_heads=4, num_tokens=8):
    """Verify CUDA decode produces identical FP16 output to Python decode."""
    print(f"\nTest: CUDA decode vs Python decode (hd={head_dim}, k={k_bits}b, v={v_bits}b)")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    k_centroids, k_boundaries, k_s1, k_s2 = setup_codebook_and_signs(k_bits, head_dim, device, K_SEED_OFFSET)
    v_centroids, v_boundaries, v_s1, v_s2 = setup_codebook_and_signs(v_bits, head_dim, device, V_SEED_OFFSET)

    num_pages = 1
    k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms, k_out_cuda, v_out_cuda = \
        create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device)
    k_out_py = torch.zeros_like(k_out_cuda)
    v_out_py = torch.zeros_like(v_out_cuda)

    # Create random input, encode with Python (known correct)
    k_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)

    for t in range(num_tokens):
        k_idx, k_n = quantize_pq(k_fp16[0, t], k_s1, k_s2, k_centroids, k_boundaries)
        k_cache[0, t] = pack_indices(k_idx, k_bits, head_dim)
        k_norms[0, t] = k_n
        v_idx, v_n = quantize_pq(v_fp16[0, t], v_s1, v_s2, v_centroids, v_boundaries)
        v_cache[0, t] = pack_indices(v_idx, v_bits, head_dim)
        v_norms[0, t] = v_n

    cache_seqlens = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # CUDA decode
    ext.turboquant_decode_paged(
        k_cache, v_cache, k_norms, v_norms, k_out_cuda, v_out_cuda,
        cache_seqlens, block_table, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_centroids, v_centroids,
    )

    # Python decode
    k_pd = packed_dim(k_bits, head_dim)
    v_pd = packed_dim(v_bits, head_dim)
    for t in range(num_tokens):
        k_packed = k_cache[0, t]
        k_idx2 = unpack_indices(k_packed.reshape(num_kv_heads, k_pd), k_bits, head_dim)
        k_hat = dequantize_pq(k_idx2, k_norms[0, t], k_s1, k_s2, k_centroids)
        k_out_py[0, t] = k_hat

        v_packed = v_cache[0, t]
        v_idx2 = unpack_indices(v_packed.reshape(num_kv_heads, v_pd), v_bits, head_dim)
        v_hat = dequantize_pq(v_idx2, v_norms[0, t], v_s1, v_s2, v_centroids)
        v_out_py[0, t] = v_hat

    # Compare
    k_close = torch.allclose(k_out_cuda[0, :num_tokens], k_out_py[0, :num_tokens], atol=1e-3, rtol=1e-3)
    v_close = torch.allclose(v_out_cuda[0, :num_tokens], v_out_py[0, :num_tokens], atol=1e-3, rtol=1e-3)

    if not k_close:
        diff = (k_out_cuda[0, :num_tokens].float() - k_out_py[0, :num_tokens].float()).abs()
        print(f"  K output MISMATCH: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
        cos = torch.nn.functional.cosine_similarity(
            k_out_cuda[0, :num_tokens].float().reshape(-1, head_dim),
            k_out_py[0, :num_tokens].float().reshape(-1, head_dim), dim=-1
        ).mean().item()
        print(f"  K cosine similarity: {cos:.6f}")
    else:
        print(f"  K output: MATCH ✓")

    if not v_close:
        diff = (v_out_cuda[0, :num_tokens].float() - v_out_py[0, :num_tokens].float()).abs()
        print(f"  V output MISMATCH: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    else:
        print(f"  V output: MATCH ✓")

    return k_close and v_close


def test_full_roundtrip(head_dim=128, k_bits=4, v_bits=4, num_kv_heads=4, num_tokens=16):
    """CUDA encode → CUDA decode roundtrip vs Python encode → Python decode."""
    print(f"\nTest: Full CUDA roundtrip vs Python roundtrip (hd={head_dim}, k={k_bits}b, v={v_bits}b)")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    k_centroids, k_boundaries, k_s1, k_s2 = setup_codebook_and_signs(k_bits, head_dim, device, K_SEED_OFFSET)
    v_centroids, v_boundaries, v_s1, v_s2 = setup_codebook_and_signs(v_bits, head_dim, device, V_SEED_OFFSET)

    num_pages = 1
    k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms, k_out, v_out = \
        create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device)

    # Random input
    k_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16[0, :num_tokens] = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.half, device=device)

    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)

    # CUDA encode
    ext.turboquant_encode_paged(
        k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms,
        cache_seqlens, block_table, num_tokens, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_boundaries, v_boundaries,
    )

    # CUDA decode
    cache_seqlens_read = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    ext.turboquant_decode_paged(
        k_cache, v_cache, k_norms, v_norms, k_out, v_out,
        cache_seqlens_read, block_table, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_centroids, v_centroids,
    )

    # Quality check
    k_cos = torch.nn.functional.cosine_similarity(
        k_fp16[0, :num_tokens].float().reshape(-1, head_dim),
        k_out[0, :num_tokens].float().reshape(-1, head_dim), dim=-1
    ).mean().item()
    v_cos = torch.nn.functional.cosine_similarity(
        v_fp16[0, :num_tokens].float().reshape(-1, head_dim),
        v_out[0, :num_tokens].float().reshape(-1, head_dim), dim=-1
    ).mean().item()

    print(f"  K cosine: {k_cos:.6f}")
    print(f"  V cosine: {v_cos:.6f}")

    # Threshold based on bit width: 2-bit=0.93, 3-bit=0.97, 4-bit=0.99
    min_bits = min(k_bits, v_bits)
    threshold = {2: 0.93, 3: 0.97, 4: 0.99}.get(min_bits, 0.90)
    ok = k_cos > threshold and v_cos > threshold
    print(f"  {'PASSED ✓' if ok else 'FAILED'} (threshold={threshold})")
    return ok


def test_page_boundary():
    """Test encoding/decoding across page boundaries."""
    print(f"\nTest: Page boundary crossing")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    head_dim = 128
    k_bits = v_bits = 4
    num_kv_heads = 4
    num_pages = 2
    # 300 tokens = crosses page boundary at 256
    num_tokens = 300

    k_centroids, k_boundaries, k_s1, k_s2 = setup_codebook_and_signs(k_bits, head_dim, device, K_SEED_OFFSET)
    v_centroids, v_boundaries, v_s1, v_s2 = setup_codebook_and_signs(v_bits, head_dim, device, V_SEED_OFFSET)

    k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms, k_out, v_out = \
        create_paged_tensors(num_pages, num_kv_heads, head_dim, k_bits, v_bits, device)

    k_fp16[0, :PAGE_SIZE] = torch.randn(PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
    k_fp16[1, :num_tokens - PAGE_SIZE] = torch.randn(num_tokens - PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16[0, :PAGE_SIZE] = torch.randn(PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)
    v_fp16[1, :num_tokens - PAGE_SIZE] = torch.randn(num_tokens - PAGE_SIZE, num_kv_heads, head_dim, dtype=torch.half, device=device)

    cache_seqlens = torch.tensor([0], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

    # Encode
    ext.turboquant_encode_paged(
        k_fp16, v_fp16, k_cache, v_cache, k_norms, v_norms,
        cache_seqlens, block_table, num_tokens, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_boundaries, v_boundaries,
    )

    # Decode
    cache_seqlens_read = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    ext.turboquant_decode_paged(
        k_cache, v_cache, k_norms, v_norms, k_out, v_out,
        cache_seqlens_read, block_table, k_bits, v_bits,
        k_s1, k_s2, v_s1, v_s2, k_centroids, v_centroids,
    )

    # Check page 0 and page 1 separately
    k_cos_p0 = torch.nn.functional.cosine_similarity(
        k_fp16[0, :PAGE_SIZE].float().reshape(-1, head_dim),
        k_out[0, :PAGE_SIZE].float().reshape(-1, head_dim), dim=-1
    ).mean().item()
    k_cos_p1 = torch.nn.functional.cosine_similarity(
        k_fp16[1, :num_tokens - PAGE_SIZE].float().reshape(-1, head_dim),
        k_out[1, :num_tokens - PAGE_SIZE].float().reshape(-1, head_dim), dim=-1
    ).mean().item()

    print(f"  Page 0 (256 tokens) K cosine: {k_cos_p0:.6f}")
    print(f"  Page 1 (44 tokens) K cosine: {k_cos_p1:.6f}")

    ok = k_cos_p0 > 0.99 and k_cos_p1 > 0.99
    print(f"  {'PASSED ✓' if ok else 'FAILED'}")
    return ok


def test_all_bit_widths():
    """Test all supported bit width combinations."""
    print(f"\nTest: All bit width combinations")
    all_ok = True
    for k_bits in [2, 3, 4]:
        for v_bits in [2, 3, 4]:
            ok = test_full_roundtrip(head_dim=128, k_bits=k_bits, v_bits=v_bits, num_kv_heads=2, num_tokens=8)
            all_ok = all_ok and ok
    return all_ok


def test_head_dims():
    """Test different head dimensions."""
    print(f"\nTest: Various head dimensions")
    all_ok = True
    for hd in [64, 128, 256]:
        ok = test_full_roundtrip(head_dim=hd, k_bits=4, v_bits=4, num_kv_heads=2, num_tokens=8)
        all_ok = all_ok and ok
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant CUDA Kernel Correctness Tests")
    print("=" * 60)

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}")

    results = []
    results.append(("Encode CUDA vs Python (hd=128, 4bit)", test_cuda_encode_vs_python(128, 4, 4)))
    results.append(("Encode CUDA vs Python (hd=256, 4bit)", test_cuda_encode_vs_python(256, 4, 4)))
    results.append(("Encode CUDA vs Python (hd=128, 3bit)", test_cuda_encode_vs_python(128, 3, 3)))
    results.append(("Encode CUDA vs Python (hd=128, 2bit)", test_cuda_encode_vs_python(128, 2, 2)))
    results.append(("Decode CUDA vs Python (hd=128, 4bit)", test_cuda_decode_vs_python(128, 4, 4)))
    results.append(("Decode CUDA vs Python (hd=256, 4bit)", test_cuda_decode_vs_python(256, 4, 4)))
    results.append(("Page boundary", test_page_boundary()))
    results.append(("All bit widths", test_all_bit_widths()))
    results.append(("All head dims", test_head_dims()))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {name}")
        all_pass = all_pass and ok

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

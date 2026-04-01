"""
Unit tests for TurboQuant PolarQuant implementation.
Standalone - imports turboquant_metadata directly.
"""

import sys, os, math
import torch
import torch.testing

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "cache")
import importlib.util

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

tq = load_mod("turboquant_metadata", os.path.join(CACHE_DIR, "turboquant_metadata.py"))


def test_codebook():
    print("Test: codebook")
    for bits in [1, 2, 3, 4]:
        for dim in [64, 128]:
            c, b = tq.compute_codebook(bits, dim)
            n = 1 << bits
            assert len(c) == n, f"Expected {n} centroids, got {len(c)}"
            assert len(b) == n - 1
            assert c == sorted(c), "Centroids not sorted"
            # Symmetric
            assert abs(c[0] + c[-1]) < 0.01, f"Not symmetric: {c[0]}, {c[-1]}"
            print(f"  bits={bits} dim={dim}: {[f'{v:.4f}' for v in c]}")
    print("  PASSED")


def test_wht_signs():
    print("\nTest: WHT sign determinism")
    s1a, s2a = tq.generate_wht_signs(128, 42)
    s1b, s2b = tq.generate_wht_signs(128, 42)
    assert torch.equal(s1a, s1b) and torch.equal(s2a, s2b), "Not deterministic"
    s1c, s2c = tq.generate_wht_signs(128, 43)
    assert not torch.equal(s1a, s1c), "Different seeds gave same signs"
    assert all(v in (-1.0, 1.0) for v in s1a.tolist()), "Signs not ±1"
    print("  PASSED")


def test_fwht():
    print("\nTest: FWHT properties")
    for dim in [32, 64, 128]:
        x = torch.randn(4, dim)
        y = tq._fwht_pow2(x)
        # WHT is its own inverse (up to normalization): H(H(x)) = x
        x_back = tq._fwht_pow2(y)
        torch.testing.assert_close(x, x_back, atol=1e-5, rtol=1e-5)
        # Preserves L2 norm (orthonormal)
        x_norm = x.norm(dim=-1)
        y_norm = y.norm(dim=-1)
        torch.testing.assert_close(x_norm, y_norm, atol=1e-4, rtol=1e-4)
        print(f"  dim={dim}: H(H(x))=x ✓, ||Hx||=||x|| ✓")
    print("  PASSED")


def test_pack_unpack():
    print("\nTest: pack/unpack roundtrip")
    for bits in [2, 3, 4]:
        for dim in [64, 128]:
            max_val = (1 << bits) - 1
            idx = torch.randint(0, max_val + 1, (8, dim), dtype=torch.uint8)
            packed = tq.pack_indices(idx, bits, dim)
            unpacked = tq.unpack_indices(packed, bits, dim)
            assert torch.equal(idx, unpacked), f"Roundtrip failed bits={bits} dim={dim}"
            pd = tq.packed_dim(bits, dim)
            assert packed.shape[-1] == pd
            print(f"  bits={bits} dim={dim}: {dim}→{pd} bytes ✓")
    print("  PASSED")


def test_quantize_dequantize():
    print("\nTest: PolarQuant roundtrip")
    torch.manual_seed(42)
    for bits in [2, 3, 4]:
        for dim in [64, 128]:
            c, b = tq.compute_codebook(bits, dim)
            centroids = torch.tensor(c, dtype=torch.float32)
            boundaries = torch.tensor(b, dtype=torch.float32)
            s1, s2 = tq.generate_wht_signs(dim, 42)

            x = torch.randn(32, dim)
            indices, norms = tq.quantize_pq(x, s1, s2, centroids, boundaries)
            x_hat = tq.dequantize_pq(indices, norms, s1, s2, centroids)

            cos = torch.nn.functional.cosine_similarity(x, x_hat.float(), dim=-1).mean().item()
            mse = ((x - x_hat.float()) ** 2).mean().item()
            rel = mse / (x ** 2).mean().item()

            print(f"  bits={bits} dim={dim}: cos={cos:.4f} rel_err={rel:.4f}")
            assert cos > 0.90, f"Cosine too low: {cos}"
    print("  PASSED")


def test_full_pipeline():
    """Test pack→unpack→dequant pipeline (what CacheLayer does)."""
    print("\nTest: full pipeline (pack indices → unpack → dequant)")
    torch.manual_seed(42)
    for bits in [2, 3, 4]:
        dim = 128
        c, b = tq.compute_codebook(bits, dim)
        centroids = torch.tensor(c, dtype=torch.float32)
        boundaries = torch.tensor(b, dtype=torch.float32)
        s1, s2 = tq.generate_wht_signs(dim, 42)

        x = torch.randn(16, dim)
        indices, norms = tq.quantize_pq(x, s1, s2, centroids, boundaries)

        # Pack and unpack
        packed = tq.pack_indices(indices, bits, dim)
        indices2 = tq.unpack_indices(packed, bits, dim)
        assert torch.equal(indices, indices2), "Pack/unpack corrupted indices"

        # Dequant from unpacked
        x_hat = tq.dequantize_pq(indices2, norms, s1, s2, centroids)
        cos = torch.nn.functional.cosine_similarity(x, x_hat.float(), dim=-1).mean().item()
        print(f"  bits={bits}: cos={cos:.4f} ✓")
        assert cos > 0.90

    print("  PASSED")


def test_gpu():
    if not torch.cuda.is_available():
        print("\nTest: GPU - SKIPPED")
        return

    print("\nTest: GPU roundtrip")
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    dim = 128
    bits = 4
    c, b = tq.compute_codebook(bits, dim)
    centroids = torch.tensor(c, dtype=torch.float32, device=device)
    boundaries = torch.tensor(b, dtype=torch.float32, device=device)
    s1, s2 = tq.generate_wht_signs(dim, 42)
    s1, s2 = s1.to(device), s2.to(device)

    x = torch.randn(64, dim, device=device)
    indices, norms = tq.quantize_pq(x, s1, s2, centroids, boundaries)
    packed = tq.pack_indices(indices, bits, dim)
    indices2 = tq.unpack_indices(packed, bits, dim)
    x_hat = tq.dequantize_pq(indices2, norms, s1, s2, centroids)

    cos = torch.nn.functional.cosine_similarity(x, x_hat.float(), dim=-1).mean().item()
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  4-bit dim=128: cos={cos:.4f}")
    assert cos > 0.95
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("TurboQuant Unit Tests")
    print("=" * 60)

    test_codebook()
    test_wht_signs()
    test_fwht()
    test_pack_unpack()
    test_quantize_dequantize()
    test_full_pipeline()
    test_gpu()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

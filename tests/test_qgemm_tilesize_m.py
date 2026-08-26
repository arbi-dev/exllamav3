# The multi-row-block GEMM shapes (TILESIZE_M > 16) must be a pure scheduling change: a row
# block's accumulation order over k is whatever the 16-row shape with the same TILESIZE_K /
# TILESIZE_N / stage counts already does, so the outputs have to match BIT FOR BIT, not within
# a tolerance. Shape 5 (32, 32, 128) is shape 2 (16, 32, 128) with a second row block, so that
# pair is the controlled comparison; a tolerance-based check here would pass straight through
# a reduction-order change, which is the failure this guards.
#
# The reduction staging area is per row block for the same reason, and getting that wrong
# showed up only at some bit widths and only above 24 rows, so both axes are swept.

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

SHAPE_M16 = 2
SHAPE_M32 = 5

# Spans the first row block's full/partial split (17..31), both blocks full (32), and the
# strip boundary above it (33..)
ROWS = [17, 20, 24, 25, 26, 31, 32, 33, 48, 64, 96, 127, 128, 144, 160]
SIZES = [(5120, 5120), (5120, 13824), (2048, 7168)]


def _operands(k, n, bits, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(
        -32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16, generator=g
    ).to(device)
    suh = (torch.randn(k, generator=g) * 0.1 + 1.0).half().to(device)
    svh = (torch.randn(n, generator=g) * 0.1 + 1.0).half().to(device)
    a = (torch.randn((max(ROWS), k), generator=g) * 0.05).half().to(device)
    return trellis, suh, svh, a


def _gemm(a, trellis, suh, svh, shape_idx, c_dtype, n):
    a = a.contiguous()
    a_had = torch.empty_like(a)
    c = torch.empty((a.shape[0], n), dtype=c_dtype, device=a.device)
    ext.exl3_gemm(a, trellis, c, suh, a_had, svh, shape_idx, False, False, 0)
    return c


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    ext.exl3_gemm_num_kernel_shapes() < SHAPE_M32, reason="no multi-row-block shapes built"
)
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("size", SIZES)
@torch.inference_mode()
def test_tilesize_m32_matches_m16_bitwise(bits, size, device="cuda:0"):
    k, n = size
    trellis, suh, svh, a_all = _operands(k, n, bits, 11 + bits, device)
    for rows in ROWS:
        a = a_all[:rows]
        ref = _gemm(a, trellis, suh, svh, SHAPE_M16, torch.float16, n)
        got = _gemm(a, trellis, suh, svh, SHAPE_M32, torch.float16, n)
        torch.cuda.synchronize()
        assert torch.equal(ref, got), f"bits={bits} {k}x{n} rows={rows}: not bit-identical"
        # A race in the staging area shows up as run-to-run drift, which a single
        # comparison against a same-run reference can miss
        again = _gemm(a, trellis, suh, svh, SHAPE_M32, torch.float16, n)
        torch.cuda.synchronize()
        assert torch.equal(got, again), f"bits={bits} {k}x{n} rows={rows}: not reproducible"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(
    ext.exl3_gemm_num_kernel_shapes() < SHAPE_M32, reason="no multi-row-block shapes built"
)
@torch.inference_mode()
def test_tilesize_m32_matches_m16_bitwise_fp32_out(device="cuda:0"):
    k, n, bits = 5120, 5120, 4
    trellis, suh, svh, a_all = _operands(k, n, bits, 7, device)
    for rows in [17, 32, 33, 64, 144]:
        a = a_all[:rows]
        ref = _gemm(a, trellis, suh, svh, SHAPE_M16, torch.float32, n)
        got = _gemm(a, trellis, suh, svh, SHAPE_M32, torch.float32, n)
        torch.cuda.synchronize()
        assert torch.equal(ref, got), f"rows={rows}: fp32 output not bit-identical"


@pytest.mark.skipif(
    ext.exl3_gemm_num_kernel_shapes() < SHAPE_M32, reason="no multi-row-block shapes built"
)
def test_multi_row_block_shapes_declined_at_or_below_16_rows():
    # The <=16-row path is the hottest in the engine and must keep its existing shape set:
    # a 32-row shape there doubles the MMA count for no decode saving
    for shape_idx in range(1, ext.exl3_gemm_num_kernel_shapes() + 1):
        for rows in (1, 5, 8, 15, 16):
            compat = ext.exl3_gemm_shape_compat(shape_idx, rows, 5120, 5120, 4)
            if shape_idx >= SHAPE_M32:
                assert not compat, f"shape {shape_idx} offered at {rows} rows"
    assert ext.exl3_gemm_shape_compat(SHAPE_M32, 17, 5120, 5120, 4)

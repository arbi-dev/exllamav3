"""Caller-owned native GEMM locks: exact sizing, guards, concurrency and capture.

CPU proof compiles the actual bound against every shipped shape. CUDA cases use
forced native kernels, so decoder GEMV/INT8 dispatch cannot hide shared-lock use.
"""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "exllamav3" / "exllamav3_ext"


def test_lock_bound_covers_all_compiled_shapes(tmp_path):
    # exl3_kernel_map.cuh includes cuda_fp16.h and declares __host__ __device__ helpers, so the
    # metadata is compiled as CUDA (host pass only, no device code is instantiated)
    compiler = shutil.which("nvcc") or (shutil.which("/usr/local/cuda/bin/nvcc"))
    if compiler is None:
        pytest.skip("needs nvcc")
    # Build actual metadata and the production sizing helper, with no CUDA or
    # extension load. Guard every supported TILESIZE_N, including future shapes.
    source = tmp_path / "lock_bound.cu"
    source.write_text(r'''
#include "quant/exl3_gemm_lock_size.h"
#include "quant/exl3_kernel_map.cuh"
#include <initializer_list>
constexpr bool covers_all_shapes() {
    constexpr int widths[] = {EXL3_GEMM_TILESIZE_N};
    static_assert(sizeof(widths) / sizeof(int) == EXL3_GEMM_NUM_SHAPES + 1);
    for (int shape = 1; shape <= EXL3_GEMM_NUM_SHAPES; ++shape) {
        if (widths[shape] < 128 || widths[shape] % 128 != 0) return false;
        for (int n : {128, 1152, 4352, 4608, 5120, 248320}) {
            const auto count = exl3_gemm_lock_count(n);
            if (count < n / widths[shape]) return false;
        }
    }
    return true;
}
static_assert(covers_all_shapes());
static_assert(exl3_gemm_lock_count(5120) * sizeof(int) == 160);
static_assert(exl3_gemm_lock_count(248320) == 1940);
''')
    subprocess.run([compiler, "-std=c++17", "-c", "-I", str(EXT_DIR), str(source), "-o", str(tmp_path / "lock_bound.o")],
                   check=True)


@pytest.fixture(scope="module")
def ext():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA and the patched native extension")
    from exllamav3.ext import exllamav3_ext
    assert hasattr(exllamav3_ext, "exl3_gemm_lock_elements"), "extension must be rebuilt"
    return exllamav3_ext


def operands(m, k, n, bits, seed=27):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits),
                           dtype=torch.int16, generator=generator).cuda()
    a = (torch.randn((m, k), generator=generator) * 0.05).half().cuda()
    suh = (torch.randn(k, generator=generator) * 0.1 + 1).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.1 + 1).half().cuda()
    return (a, trellis, torch.empty((m, n), device="cuda", dtype=torch.float32),
            suh, torch.empty_like(a), svh)


def launch(ext, args, shape=2, sms=64, locks=None, n_out=0):
    return ext.exl3_gemm(*args, shape, False, True, sms, n_out, locks)


@pytest.mark.parametrize("shape", range(1, 12))
@pytest.mark.parametrize("n_out", [0, 1024])
def test_exact_bound_matches_legacy_and_preserves_canary(ext, shape, n_out):
    args = operands(97, 5120, 5120, 6)
    width = n_out or 5120
    if not ext.exl3_gemm_shape_compat(shape, 97, 5120, width, 6):
        pytest.skip("shape does not support this geometry")
    if n_out:
        args = (*args[:2], torch.empty((97, n_out), device="cuda", dtype=torch.float32), *args[3:])
    count = ext.exl3_gemm_lock_elements(width)
    storage = torch.full((count + 16,), 2147483647, device="cuda", dtype=torch.int32)
    owned = storage[:count]
    owned.zero_()
    launch(ext, args, shape=shape, locks=None, n_out=n_out)
    ref = args[2].clone()
    launch(ext, args, shape=shape, locks=owned, n_out=n_out)
    torch.cuda.synchronize()
    assert torch.equal(args[2], ref)
    assert torch.isfinite(ref).all() and ref.abs().max() > 0
    assert torch.count_nonzero(owned) == 0
    assert torch.all(storage[count:] == 2147483647)


def test_owned_lock_validation(ext):
    args = operands(8, 5120, 1024, 4)
    for width in [0, -128, 127, 129]:
        with pytest.raises(RuntimeError, match="positive multiple of 128"):
            ext.exl3_gemm_lock_elements(width)
    count = ext.exl3_gemm_lock_elements(1024)
    invalid = [
        (torch.zeros(count, dtype=torch.int32), "CUDA tensor"),
        (torch.zeros(count, device="cuda", dtype=torch.int64), "int32"),
        (torch.zeros((2, count), device="cuda", dtype=torch.int32), "one-dimensional"),
        (torch.zeros(count * 2, device="cuda", dtype=torch.int32)[::2], "contiguous"),
        (torch.zeros(count - 1, device="cuda", dtype=torch.int32), "too small"),
    ]
    for locks, reason in invalid:
        with pytest.raises(RuntimeError, match=reason):
            launch(ext, args, locks=locks)
    if torch.cuda.device_count() > 1:
        with pytest.raises(RuntimeError, match="A's device"):
            launch(ext, args, locks=torch.zeros(count, device="cuda:1", dtype=torch.int32))


@pytest.mark.parametrize("captured", [False, True])
def test_vision_native_overlaps_text_native_with_owned_locks(ext, captured):
    if torch.cuda.get_device_properties(0).multi_processor_count < 100:
        pytest.skip("36+64 resident blocks need at least100 SMs for this regression")
    vision = operands(4096, 1152, 1152, 6, seed=33)
    text = operands(8, 5120, 1024, 4, seed=34)
    locks = torch.zeros(ext.exl3_gemm_lock_elements(1152), device="cuda", dtype=torch.int32)
    vision_stream, text_stream = torch.cuda.Stream(), torch.cuda.Stream()
    launch(ext, vision, shape=11, sms=36, locks=locks)
    vision_ref = vision[2].clone()
    launch(ext, text, shape=2, sms=64)
    text_ref = text[2].clone()
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=vision_stream):
            launch(ext, vision, shape=11, sms=36, locks=locks)
    for _ in range(10):
        outputs = []
        for _ in range(4):
            with torch.cuda.stream(vision_stream):
                if graph is None:
                    launch(ext, vision, shape=11, sms=36, locks=locks)
                else:
                    graph.replay()
                vision_copy = vision[2].clone()
            with torch.cuda.stream(text_stream):
                launch(ext, text, shape=2, sms=64)
                text_copy = text[2].clone()
            outputs.append((vision_copy, text_copy))
        torch.cuda.synchronize()
        for vision_copy, text_copy in outputs:
            assert torch.equal(vision_copy, vision_ref)
            assert torch.equal(text_copy, text_ref)
    assert torch.count_nonzero(locks) == 0


@pytest.mark.parametrize("rows", [1, 2])
def test_owned_locks_do_not_enter_device_global_int8_workspace(ext, monkeypatch, rows):
    monkeypatch.setenv("EXL3_INT8_GEMV", "2")
    args = operands(rows, 5120, 1024, 4)
    # A nonzero output width excludes the experimental INT8 path for the serial
    # reference. The owned path must provide the same isolation even when the
    # public API's default output-width argument is used. No unsafe overlap is
    # needed to exercise this dispatch contract.
    assert launch(ext, args, shape=2, sms=64, n_out=1024) == 2
    reference = args[2].clone()
    locks = torch.zeros(ext.exl3_gemm_lock_elements(1024), device="cuda", dtype=torch.int32)
    assert launch(ext, args, shape=2, sms=64, locks=locks) == 2
    torch.cuda.synchronize()
    assert torch.equal(args[2], reference)
    assert torch.count_nonzero(locks) == 0

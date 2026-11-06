#pragma once
#include <cstdint>

// Regular GEMM reuses its counters between serial M strips. The inner kernel
// statically requires TILESIZE_N to be a multiple of 128 and indexes only
// locks[slice2_n], with slice2_n < size_n / TILESIZE_N. This does not size
// the separate multi-GEMM / MoE barrier and scheduler workspace.
// Public callers validate a positive, 128-aligned output width first.
constexpr int64_t exl3_gemm_lock_count(int64_t size_n)
{
    return size_n / 128;
}

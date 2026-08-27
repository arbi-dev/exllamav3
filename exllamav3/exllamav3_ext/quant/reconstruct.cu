#include <cuda_fp16.h>
#include "reconstruct.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Reconstruct encoded+packed tensor
*/
void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);
    TORCH_CHECK_DTYPE(unpacked, kHalf);

    int rows = packed.size(0);
    int packed_cols = packed.size(1);

    if (unpacked.numel() == 0)
        return;

    TORCH_CHECK(unpacked.size(1) % 128 == 0, "unpacked N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");

    int cols = unpacked.size(1) / 16;
    int packed_n_offset = n_offset / 16;

    dim3 blockDim(256);
    dim3 gridDim(cols / 8, rows);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    reconstruct_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>
    (
        (half*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        packed_cols,
        packed_n_offset
    );
    cuda_check(cudaPeekAtLastError());
}


// Fused reconstruct + both-side Hadamard: emits W = diag(suh) . H128 . W_hat . H128 . diag(svh)
// (per 128x128 tile, 1/sqrt(128) per side), i.e. the ORIGINAL-basis weights, so the hgemm
// prefill path can run on the raw input with no standalone input/output had_r_128 launches.
//
// Both transforms accumulate in fp32 and round to fp16 exactly once per pass, at the tile
// store and at the global store. This weight feeds a prefill GEMM whose output lands in the
// KV that the MTP drafter reads, a channel sensitive to weight error far below the 4-bit
// quantisation floor, so the butterfly cannot afford a rounding per stage. The tile itself
// stays fp16, so shared memory and occupancy are unchanged.
//
// 128x128 tile lives in shared memory with an XOR swizzle on the half4 chunk index
// (chunk ^= (row >> 2) & 31) making all phases bank-conflict-free except the 16-lane dequant
// scatter; both transforms reuse the natural-order Sylvester butterfly (symmetric, so
// column-then-row order is H W H).
#define RH_THREADS 256

template <int K, int cb>
__global__ __launch_bounds__(RH_THREADS)
void reconstruct_had_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        register FragB frag[2];
        dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        // c[i]: 4 columns x row (4 * lane + i), all four rows share swizzle == lane
        float4 c[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            float2 lo = __half22float2(v.x), hi = __half22float2(v.y);
            c[i] = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        // butterfly over the 4 rows, four columns at a time; the half4 store below is this
        // pass's only rounding
        {
            float4 s0 = make_float4(c[0].x + c[1].x, c[0].y + c[1].y,
                                    c[0].z + c[1].z, c[0].w + c[1].w);
            float4 d0 = make_float4(c[0].x - c[1].x, c[0].y - c[1].y,
                                    c[0].z - c[1].z, c[0].w - c[1].w);
            float4 s1 = make_float4(c[2].x + c[3].x, c[2].y + c[3].y,
                                    c[2].z + c[3].z, c[2].w + c[3].w);
            float4 d1 = make_float4(c[2].x - c[3].x, c[2].y - c[3].y,
                                    c[2].z - c[3].z, c[2].w - c[3].w);
            c[0] = make_float4((s0.x + s1.x) * r_scale, (s0.y + s1.y) * r_scale,
                               (s0.z + s1.z) * r_scale, (s0.w + s1.w) * r_scale);
            c[1] = make_float4((d0.x + d1.x) * r_scale, (d0.y + d1.y) * r_scale,
                               (d0.z + d1.z) * r_scale, (d0.w + d1.w) * r_scale);
            c[2] = make_float4((s0.x - s1.x) * r_scale, (s0.y - s1.y) * r_scale,
                               (s0.z - s1.z) * r_scale, (s0.w - s1.w) * r_scale);
            c[3] = make_float4((d0.x - d1.x) * r_scale, (d0.y - d1.y) * r_scale,
                               (d0.z - d1.z) * r_scale, (d0.w - d1.w) * r_scale);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            shuffle_had_f4x32(c[i].x, c[i].y, c[i].z, c[i].w, lane_id);
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = __floats2half2_rn(c[i].x, c[i].y);
            v.y = __floats2half2_rn(c[i].z, c[i].w);
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    // Row transform fused with the store: after the lane butterfly, lane l holds the
    // FINAL columns 4l..4l+3 of row R. Apply the sign scales in registers and write the
    // coalesced 256-byte row directly.
    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    const float2 sv01 = __half22float2(sv4.x);
    const float2 sv23 = __half22float2(sv4.y);
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        float h0 = (s0 + s1) * r_scale, h1 = (d0 + d1) * r_scale;
        float h2 = (s0 - s1) * r_scale, h3 = (d0 - d1) * r_scale;
        shuffle_had_f4x32(h0, h1, h2, h3, lane_id);
        float su = __half2float(suh[kb * 128 + R]);
        half4 v;
        v.x = __floats2half2_rn(h0 * su * sv01.x, h1 * su * sv01.y);
        v.y = __floats2half2_rn(h2 * su * sv23.x, h3 * su * sv23.y);
        *((half4*) (g_unpacked + (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4)) = v;
    }
}


#define __(i, cb) reconstruct_had_kernel<i, cb>
constexpr auto reconstruct_had_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Reconstruct in the original (unrotated) basis: fused suh/H/W_hat/H/svh. Both dims must be
multiples of 128. n_offset slices along n; svh must be pre-offset by the caller.
*/
void reconstruct_had_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    at::Tensor suh,
    at::Tensor svh,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);
    TORCH_CHECK_DTYPE(unpacked, kHalf);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);

    if (unpacked.numel() == 0)
        return;

    TORCH_CHECK(unpacked.size(0) % 128 == 0, "reconstruct_had: K dimension must be divisible by 128");
    TORCH_CHECK(unpacked.size(1) % 128 == 0, "reconstruct_had: N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");
    TORCH_CHECK(suh.numel() >= unpacked.size(0), "reconstruct_had: suh size");
    TORCH_CHECK(svh.numel() >= unpacked.size(1), "reconstruct_had: svh size");

    dim3 blockDim(256);
    dim3 gridDim(unpacked.size(1) / 128, unpacked.size(0) / 128);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    reconstruct_had_kernel_instances[cbi]<<<gridDim, RH_THREADS, 0, stream>>>
    (
        (half*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        (const half*) suh.data_ptr(),
        (const half*) svh.data_ptr(),
        packed.size(1),
        (int) (n_offset / 16)
    );
    cuda_check(cudaPeekAtLastError());
}

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    TORCH_CHECK_SHAPES(unpacked, 1, packed, 1, 16);
    reconstruct_slice(unpacked, packed, K, mcg, mul1, 0);
}

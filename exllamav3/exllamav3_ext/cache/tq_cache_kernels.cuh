#pragma once

#include <cuda_fp16.h>
#include <cstdint>

/**
 * TurboQuant (PolarQuant) CUDA kernels for exllamav3 KV cache.
 *
 * Algorithm per vector:
 *   Forward:  x -> normalize -> D1*H*D2 rotation -> searchsorted -> pack indices
 *   Inverse:  unpack -> centroid lookup -> D2*H*D1 inverse rotation -> scale by norm
 *
 * Based on turboquant-vllm (https://github.com/varjoranta/turboquant-vllm).
 *
 * Kernel design:
 *   - One threadblock per (token, kv_head) pair
 *   - blockDim.x >= head_dim, rounded to warp multiple
 *   - WHT butterfly in shared memory with __syncthreads between stages
 *   - Norms stored in separate tensor (not packed with indices)
 */

#define TQ_PAGE_SIZE 256
#define TQ_MAX_HEAD_DIM 256
#define TQ_MAX_CENTROIDS 16


// ── Warp-level helpers ───────────────────────────────────────────────────────

/** In-place normalized Walsh-Hadamard Transform. n must be power of 2. */
__device__ __forceinline__
void tq_wht_inplace(float* __restrict__ s, int n, int tid)
{
    for (int h = 1; h < n; h <<= 1)
    {
        __syncthreads();
        int g = tid / h;
        int p = tid % h;
        int i = g * (h << 1) + p;
        if (i + h < n && tid < (n >> 1))
        {
            float a = s[i], b = s[i + h];
            s[i] = a + b;
            s[i + h] = a - b;
        }
    }
    __syncthreads();
    float inv = rsqrtf(static_cast<float>(n));
    if (tid < n) s[tid] *= inv;
    __syncthreads();
}


/** Parallel L2 norm reduction. Returns norm broadcast to all threads. */
__device__ __forceinline__
float tq_reduce_norm(const float* __restrict__ data, int n, int tid,
                     float* __restrict__ scratch)
{
    float sq = (tid < n) ? data[tid] * data[tid] : 0.0f;
    scratch[tid] = sq;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    return sqrtf(scratch[0]);
}


/**
 * PolarQuant forward transform in shared memory.
 * On entry: smem[0..hd) contains the input vector as fp32.
 * On exit:  smem[0..hd) contains the rotated+normalized coordinates.
 * Returns the L2 norm.
 */
__device__ __forceinline__
float tq_polar_forward(
    float* __restrict__ data,
    float* __restrict__ scratch,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int hd, int tid)
{
    float norm = tq_reduce_norm(data, hd, tid, scratch);

    float inv = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    if (tid < hd) data[tid] *= inv;
    __syncthreads();

    if (tid < hd) data[tid] *= signs1[tid];
    __syncthreads();
    tq_wht_inplace(data, hd, tid);
    if (tid < hd) data[tid] *= signs2[tid];
    __syncthreads();

    return norm;
}


/** PolarQuant inverse: centroid lookup is done before calling this.
 *  On entry: smem[0..hd) contains centroid values.
 *  On exit:  smem[0..hd) contains the reconstructed vector (before norm scaling). */
__device__ __forceinline__
void tq_polar_inverse(
    float* __restrict__ smem,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int hd, int tid, float norm)
{
    if (tid < hd) smem[tid] *= signs2[tid];
    __syncthreads();
    tq_wht_inplace(smem, hd, tid);
    if (tid < hd) smem[tid] *= signs1[tid];
    __syncthreads();

    if (tid < hd) smem[tid] *= norm;
    __syncthreads();
}


/** Searchsorted: find centroid index for a value given sorted boundaries. */
__device__ __forceinline__
int tq_searchsorted(float val, const float* __restrict__ boundaries, int num_boundaries)
{
    int idx = 0;
    #pragma unroll
    for (int b = 0; b < TQ_MAX_CENTROIDS - 1; b++)
    {
        if (b < num_boundaries && val > boundaries[b]) idx = b + 1;
    }
    return idx;
}


// ── Encode kernel: FP16 -> packed indices + norms ────────────────────────────

/**
 * Fused encode for paged KV cache.
 * Grid:  (seq_len * num_kv_heads, 1, batch_size)
 * Block: (threads,) where threads >= head_dim, rounded to 32
 *
 * Processes both K and V in the same threadblock.
 */
__global__ void tq_encode_paged_kernel(
    const half* __restrict__ key,           // (num_pages, PAGE_SIZE, num_kv_heads, head_dim)
    const half* __restrict__ value,
    uint8_t* __restrict__ key_cache,        // (num_pages, PAGE_SIZE, num_kv_heads, k_packed_dim)
    uint8_t* __restrict__ value_cache,
    float* __restrict__ k_norms,            // (num_pages, PAGE_SIZE, num_kv_heads)
    float* __restrict__ v_norms,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int pages_per_seq,
    int num_kv_heads,
    int head_dim,
    int k_packed_dim,
    int v_packed_dim,
    int k_bit_width,
    int v_bit_width,
    int seq_len,                            // number of new tokens
    const float* __restrict__ k_signs1,
    const float* __restrict__ k_signs2,
    const float* __restrict__ v_signs1,
    const float* __restrict__ v_signs2,
    const float* __restrict__ k_boundaries, // (num_k_centroids - 1,)
    const float* __restrict__ v_boundaries,
    int k_num_boundaries,
    int v_num_boundaries
)
{
    int pair_id = blockIdx.x;
    int batch_idx = blockIdx.z;
    int token_offset = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;
    if (token_offset >= seq_len) return;

    int tid = threadIdx.x;
    int hd = head_dim;

    // Compute paged address
    int seq_start = cache_seqlens[batch_idx];
    int token_idx = seq_start + token_offset;
    int page = token_idx / TQ_PAGE_SIZE;
    int page_off = token_idx % TQ_PAGE_SIZE;
    int mapped_page = block_table[batch_idx * pages_per_seq + page];

    // Linear index into paged tensors
    int page_stride = TQ_PAGE_SIZE * num_kv_heads;
    int fp16_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * hd;
    int k_cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * k_packed_dim;
    int v_cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * v_packed_dim;
    int norm_base = mapped_page * page_stride + page_off * num_kv_heads + head_idx;

    extern __shared__ float smem[];
    float* data = smem;
    float* scratch = smem + hd;

    // ── Encode K ──
    if (tid < hd) data[tid] = __half2float(key[fp16_base + tid]);
    else if (tid < blockDim.x) scratch[tid] = 0.0f;
    __syncthreads();

    float kn = tq_polar_forward(data, scratch, k_signs1, k_signs2, hd, tid);
    if (tid == 0) k_norms[norm_base] = kn;

    // Pack K indices
    if (k_bit_width == 4 && tid < hd / 2)
    {
        uint8_t lo = static_cast<uint8_t>(tq_searchsorted(data[tid * 2], k_boundaries, k_num_boundaries));
        uint8_t hi = static_cast<uint8_t>(tq_searchsorted(data[tid * 2 + 1], k_boundaries, k_num_boundaries));
        key_cache[k_cache_base + tid] = lo | (hi << 4);
    }
    else if (k_bit_width == 3 && tid < hd)
    {
        key_cache[k_cache_base + tid] = static_cast<uint8_t>(tq_searchsorted(data[tid], k_boundaries, k_num_boundaries));
    }
    else if (k_bit_width == 2 && tid < hd / 4)
    {
        uint8_t p = 0;
        for (int i = 0; i < 4; i++)
            p |= static_cast<uint8_t>(tq_searchsorted(data[tid * 4 + i], k_boundaries, k_num_boundaries)) << (i * 2);
        key_cache[k_cache_base + tid] = p;
    }
    __syncthreads();

    // ── Encode V ──
    if (tid < hd) data[tid] = __half2float(value[fp16_base + tid]);
    else if (tid < blockDim.x) scratch[tid] = 0.0f;
    __syncthreads();

    float vn = tq_polar_forward(data, scratch, v_signs1, v_signs2, hd, tid);
    if (tid == 0) v_norms[norm_base] = vn;

    // Pack V indices
    if (v_bit_width == 4 && tid < hd / 2)
    {
        uint8_t lo = static_cast<uint8_t>(tq_searchsorted(data[tid * 2], v_boundaries, v_num_boundaries));
        uint8_t hi = static_cast<uint8_t>(tq_searchsorted(data[tid * 2 + 1], v_boundaries, v_num_boundaries));
        value_cache[v_cache_base + tid] = lo | (hi << 4);
    }
    else if (v_bit_width == 3 && tid < hd)
    {
        value_cache[v_cache_base + tid] = static_cast<uint8_t>(tq_searchsorted(data[tid], v_boundaries, v_num_boundaries));
    }
    else if (v_bit_width == 2 && tid < hd / 4)
    {
        uint8_t p = 0;
        for (int i = 0; i < 4; i++)
            p |= static_cast<uint8_t>(tq_searchsorted(data[tid * 4 + i], v_boundaries, v_num_boundaries)) << (i * 2);
        value_cache[v_cache_base + tid] = p;
    }
}


// ── Decode kernel: packed indices + norms -> FP16 ────────────────────────────

/**
 * Dequantize from paged KV cache.
 * Grid:  (max_seq_len * num_kv_heads, 1, batch_size)
 * Block: (threads,) where threads >= head_dim, rounded to 32
 */
__global__ void tq_decode_paged_kernel(
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    const float* __restrict__ k_norms,
    const float* __restrict__ v_norms,
    half* __restrict__ k_out,               // (num_pages, PAGE_SIZE, num_kv_heads, head_dim)
    half* __restrict__ v_out,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int pages_per_seq,
    int num_kv_heads,
    int head_dim,
    int k_packed_dim,
    int v_packed_dim,
    int k_bit_width,
    int v_bit_width,
    const float* __restrict__ k_signs1,
    const float* __restrict__ k_signs2,
    const float* __restrict__ v_signs1,
    const float* __restrict__ v_signs2,
    const float* __restrict__ k_centroids,  // (num_k_centroids,)
    const float* __restrict__ v_centroids
)
{
    int pair_id = blockIdx.x;
    int batch_idx = blockIdx.z;
    int token_idx = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;

    int max_token = cache_seqlens[batch_idx];
    if (token_idx >= max_token) return;

    int tid = threadIdx.x;
    int hd = head_dim;

    int page = token_idx / TQ_PAGE_SIZE;
    int page_off = token_idx % TQ_PAGE_SIZE;
    int mapped_page = block_table[batch_idx * pages_per_seq + page];

    int page_stride = TQ_PAGE_SIZE * num_kv_heads;
    int k_cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * k_packed_dim;
    int v_cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * v_packed_dim;
    int fp16_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * hd;
    int norm_base = mapped_page * page_stride + page_off * num_kv_heads + head_idx;

    extern __shared__ float smem[];

    // ── Decode K ──
    // Unpack + centroid lookup
    if (k_bit_width == 4 && tid < hd)
    {
        int byte_idx = tid / 2;
        uint8_t packed = key_cache[k_cache_base + byte_idx];
        int idx = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        smem[tid] = k_centroids[idx];
    }
    else if (k_bit_width == 3 && tid < hd)
    {
        smem[tid] = k_centroids[key_cache[k_cache_base + tid]];
    }
    else if (k_bit_width == 2 && tid < hd)
    {
        int byte_idx = tid / 4;
        int shift = (tid % 4) * 2;
        uint8_t packed = key_cache[k_cache_base + byte_idx];
        smem[tid] = k_centroids[(packed >> shift) & 0x03];
    }
    __syncthreads();

    float kn = k_norms[norm_base];
    tq_polar_inverse(smem, k_signs1, k_signs2, hd, tid, kn);

    if (tid < hd) k_out[fp16_base + tid] = __float2half(smem[tid]);
    __syncthreads();

    // ── Decode V ──
    if (v_bit_width == 4 && tid < hd)
    {
        int byte_idx = tid / 2;
        uint8_t packed = value_cache[v_cache_base + byte_idx];
        int idx = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        smem[tid] = v_centroids[idx];
    }
    else if (v_bit_width == 3 && tid < hd)
    {
        smem[tid] = v_centroids[value_cache[v_cache_base + tid]];
    }
    else if (v_bit_width == 2 && tid < hd)
    {
        int byte_idx = tid / 4;
        int shift = (tid % 4) * 2;
        uint8_t packed = value_cache[v_cache_base + byte_idx];
        smem[tid] = v_centroids[(packed >> shift) & 0x03];
    }
    __syncthreads();

    float vn = v_norms[norm_base];
    tq_polar_inverse(smem, v_signs1, v_signs2, hd, tid, vn);

    if (tid < hd) v_out[fp16_base + tid] = __float2half(smem[tid]);
}

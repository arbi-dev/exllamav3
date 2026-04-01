#pragma once

#include <cuda_fp16.h>
#include <cstdint>

/**
 * TurboQuant (PolarQuant) CUDA kernels for exllamav3 KV cache.
 *
 * Design:
 *   - Encode: one threadblock per (token, head, K_or_V) — K and V in parallel
 *   - Decode: same parallel layout
 *   - WHT butterfly in shared memory, supports padded non-pow2 head_dim
 *   - Proper 3-bit bitstream packing (3 bits per index, not 1 byte)
 */

#define TQ_PAGE_SIZE 256
#define TQ_MAX_HEAD_DIM 256
#define TQ_MAX_CENTROIDS 16


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


__device__ __forceinline__
float tq_polar_forward(
    float* __restrict__ data,
    float* __restrict__ scratch,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int padded_dim, int tid)
{
    float norm = tq_reduce_norm(data, padded_dim, tid, scratch);

    float inv = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    if (tid < padded_dim) data[tid] *= inv;
    __syncthreads();

    if (tid < padded_dim) data[tid] *= signs1[tid];
    __syncthreads();
    tq_wht_inplace(data, padded_dim, tid);
    if (tid < padded_dim) data[tid] *= signs2[tid];
    __syncthreads();

    return norm;
}


__device__ __forceinline__
void tq_polar_inverse(
    float* __restrict__ smem,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int padded_dim, int tid, float norm)
{
    if (tid < padded_dim) smem[tid] *= signs2[tid];
    __syncthreads();
    tq_wht_inplace(smem, padded_dim, tid);
    if (tid < padded_dim) smem[tid] *= signs1[tid];
    __syncthreads();

    if (tid < padded_dim) smem[tid] *= norm;
    __syncthreads();
}


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


// ── Pack helpers ────────────────────────────────────────────────────────────

__device__ __forceinline__
void tq_pack_4bit(float* data, uint8_t* out, const float* boundaries, int num_b, int padded_dim, int tid)
{
    if (tid < padded_dim / 2)
    {
        uint8_t lo = static_cast<uint8_t>(tq_searchsorted(data[tid * 2], boundaries, num_b));
        uint8_t hi = static_cast<uint8_t>(tq_searchsorted(data[tid * 2 + 1], boundaries, num_b));
        out[tid] = lo | (hi << 4);
    }
}

__device__ __forceinline__
void tq_pack_3bit(float* data, uint8_t* out, const float* boundaries, int num_b, int padded_dim, int tid)
{
    // Proper 3-bit bitstream: each thread handles one index, writes its 3 bits
    // to the correct byte positions. We use shared memory as intermediate.
    // First: each thread computes its index
    int idx = 0;
    if (tid < padded_dim)
        idx = tq_searchsorted(data[tid], boundaries, num_b);

    __syncthreads();

    // Pack: thread 0 does serial packing from shared memory indices
    // (for correctness; optimize later with ballot_sync)
    int packed_bytes = (padded_dim * 3 + 7) / 8;
    if (tid == 0)
    {
        for (int i = 0; i < packed_bytes; i++) out[i] = 0;
    }
    __syncthreads();

    if (tid < padded_dim)
    {
        int base_bit = tid * 3;
        for (int b = 0; b < 3; b++)
        {
            int bit_pos = base_bit + b;
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            if ((idx >> b) & 1)
                atomicOr(reinterpret_cast<unsigned int*>(out + (byte_idx & ~3)),
                         1u << (bit_idx + (byte_idx & 3) * 8));
        }
    }
}

__device__ __forceinline__
void tq_pack_2bit(float* data, uint8_t* out, const float* boundaries, int num_b, int padded_dim, int tid)
{
    if (tid < padded_dim / 4)
    {
        uint8_t p = 0;
        for (int i = 0; i < 4; i++)
            p |= static_cast<uint8_t>(tq_searchsorted(data[tid * 4 + i], boundaries, num_b)) << (i * 2);
        out[tid] = p;
    }
}


// ── Encode kernel ───────────────────────────────────────────────────────────
// Grid: (seq_len * num_kv_heads * 2, 1, batch_size)
//   pair_id % 2 == 0 → K, pair_id % 2 == 1 → V
//   This gives K and V full parallelism.

__global__ void tq_encode_paged_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    float* __restrict__ k_norms,
    float* __restrict__ v_norms,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int pages_per_seq,
    int num_kv_heads,
    int head_dim,        // original (may be non-pow2)
    int padded_dim,      // next power of 2
    int k_packed_dim,
    int v_packed_dim,
    int k_bit_width,
    int v_bit_width,
    int seq_len,
    const float* __restrict__ k_signs1,
    const float* __restrict__ k_signs2,
    const float* __restrict__ v_signs1,
    const float* __restrict__ v_signs2,
    const float* __restrict__ k_boundaries,
    const float* __restrict__ v_boundaries,
    int k_num_boundaries,
    int v_num_boundaries
)
{
    int flat_id = blockIdx.x;
    int batch_idx = blockIdx.z;
    int is_value = flat_id % 2;
    int pair_id = flat_id / 2;
    int token_offset = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;
    if (token_offset >= seq_len) return;

    int tid = threadIdx.x;

    int seq_start = cache_seqlens[batch_idx];
    int token_idx = seq_start + token_offset;
    int page = token_idx / TQ_PAGE_SIZE;
    int page_off = token_idx % TQ_PAGE_SIZE;
    int mapped_page = block_table[batch_idx * pages_per_seq + page];

    int page_stride = TQ_PAGE_SIZE * num_kv_heads;
    int fp16_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * head_dim;
    int norm_base = mapped_page * page_stride + page_off * num_kv_heads + head_idx;

    extern __shared__ float smem[];
    float* data = smem;
    float* scratch = smem + padded_dim;

    const half* input = is_value ? value : key;
    const float* signs1 = is_value ? v_signs1 : k_signs1;
    const float* signs2 = is_value ? v_signs2 : k_signs2;
    const float* boundaries = is_value ? v_boundaries : k_boundaries;
    int num_b = is_value ? v_num_boundaries : k_num_boundaries;
    int bit_width = is_value ? v_bit_width : k_bit_width;
    int cache_packed_dim = is_value ? v_packed_dim : k_packed_dim;
    uint8_t* cache_out = is_value ? value_cache : key_cache;
    float* norms_out = is_value ? v_norms : k_norms;

    int cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * cache_packed_dim;

    // Load input, zero-pad to padded_dim
    if (tid < head_dim)
        data[tid] = __half2float(input[fp16_base + tid]);
    else if (tid < padded_dim)
        data[tid] = 0.0f;
    if (tid >= padded_dim && tid < blockDim.x)
        scratch[tid] = 0.0f;
    __syncthreads();

    float norm = tq_polar_forward(data, scratch, signs1, signs2, padded_dim, tid);
    if (tid == 0) norms_out[norm_base] = norm;

    if (bit_width == 4)
        tq_pack_4bit(data, cache_out + cache_base, boundaries, num_b, padded_dim, tid);
    else if (bit_width == 3)
        tq_pack_3bit(data, cache_out + cache_base, boundaries, num_b, padded_dim, tid);
    else if (bit_width == 2)
        tq_pack_2bit(data, cache_out + cache_base, boundaries, num_b, padded_dim, tid);
}


// ── Decode kernel ───────────────────────────────────────────────────────────
// Grid: (max_tokens * num_kv_heads * 2, 1, batch_size)

__global__ void tq_decode_paged_kernel(
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    const float* __restrict__ k_norms,
    const float* __restrict__ v_norms,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    const int32_t* __restrict__ cache_seqlens,
    const int32_t* __restrict__ block_table,
    int pages_per_seq,
    int num_kv_heads,
    int head_dim,
    int padded_dim,
    int k_packed_dim,
    int v_packed_dim,
    int k_bit_width,
    int v_bit_width,
    const float* __restrict__ k_signs1,
    const float* __restrict__ k_signs2,
    const float* __restrict__ v_signs1,
    const float* __restrict__ v_signs2,
    const float* __restrict__ k_centroids,
    const float* __restrict__ v_centroids
)
{
    int flat_id = blockIdx.x;
    int batch_idx = blockIdx.z;
    int is_value = flat_id % 2;
    int pair_id = flat_id / 2;
    int token_idx = pair_id / num_kv_heads;
    int head_idx = pair_id % num_kv_heads;

    int max_token = cache_seqlens[batch_idx];
    if (token_idx >= max_token) return;

    int tid = threadIdx.x;

    int page = token_idx / TQ_PAGE_SIZE;
    int page_off = token_idx % TQ_PAGE_SIZE;
    int mapped_page = block_table[batch_idx * pages_per_seq + page];

    int page_stride = TQ_PAGE_SIZE * num_kv_heads;
    int norm_base = mapped_page * page_stride + page_off * num_kv_heads + head_idx;
    int fp16_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * head_dim;

    const float* signs1 = is_value ? v_signs1 : k_signs1;
    const float* signs2 = is_value ? v_signs2 : k_signs2;
    const float* centroids = is_value ? v_centroids : k_centroids;
    const float* norms_in = is_value ? v_norms : k_norms;
    int bit_width = is_value ? v_bit_width : k_bit_width;
    int cache_pd = is_value ? v_packed_dim : k_packed_dim;
    const uint8_t* cache_in = is_value ? value_cache : key_cache;
    half* output = is_value ? v_out : k_out;

    int cache_base = (mapped_page * page_stride + page_off * num_kv_heads + head_idx) * cache_pd;

    extern __shared__ float smem[];

    // Initialize to zero (for padding region)
    if (tid < padded_dim) smem[tid] = 0.0f;
    __syncthreads();

    // Unpack + centroid lookup
    if (bit_width == 4 && tid < padded_dim)
    {
        int byte_idx = tid / 2;
        uint8_t packed = cache_in[cache_base + byte_idx];
        int idx = (tid & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
        smem[tid] = centroids[idx];
    }
    else if (bit_width == 3 && tid < padded_dim)
    {
        int base_bit = tid * 3;
        int idx = 0;
        for (int b = 0; b < 3; b++)
        {
            int bit_pos = base_bit + b;
            int byte_idx = bit_pos / 8;
            int bit_idx = bit_pos % 8;
            idx |= ((cache_in[cache_base + byte_idx] >> bit_idx) & 1) << b;
        }
        smem[tid] = centroids[idx];
    }
    else if (bit_width == 2 && tid < padded_dim)
    {
        int byte_idx = tid / 4;
        int shift = (tid % 4) * 2;
        uint8_t packed = cache_in[cache_base + byte_idx];
        smem[tid] = centroids[(packed >> shift) & 0x03];
    }
    __syncthreads();

    float norm = norms_in[norm_base];
    tq_polar_inverse(smem, signs1, signs2, padded_dim, tid, norm);

    // Write only head_dim elements (skip padding)
    if (tid < head_dim) output[fp16_base + tid] = __float2half(smem[tid]);
}

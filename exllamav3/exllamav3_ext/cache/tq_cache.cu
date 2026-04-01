#include <cuda_fp16.h>
#include "tq_cache.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "tq_cache_kernels.cuh"

void turboquant_encode_paged(
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& k_norms,
    const at::Tensor& v_norms,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int seq_len,
    int k_bit_width,
    int v_bit_width,
    const at::Tensor& k_signs1,
    const at::Tensor& k_signs2,
    const at::Tensor& v_signs1,
    const at::Tensor& v_signs2,
    const at::Tensor& k_boundaries,
    const at::Tensor& v_boundaries
)
{
    const at::cuda::OptionalCUDAGuard device_guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(key, kHalf);
    TORCH_CHECK_DTYPE(value, kHalf);
    TORCH_CHECK(key_cache.dtype() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.dtype() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(key.dim() == 4, "key must be 4D");

    int num_kv_heads = key.size(2);
    int head_dim = key.size(3);
    int k_packed_dim = key_cache.size(3);
    int v_packed_dim = value_cache.size(3);
    int bsz = block_table.size(0);
    int pages_per_seq = block_table.size(1);

    TORCH_CHECK(head_dim <= TQ_MAX_HEAD_DIM, "head_dim must be <= ", TQ_MAX_HEAD_DIM);
    TORCH_CHECK((head_dim & (head_dim - 1)) == 0, "head_dim must be power of 2");

    int bt = ((head_dim + 31) / 32) * 32;
    int smem = (head_dim + bt) * sizeof(float);

    dim3 grid(seq_len * num_kv_heads, 1, bsz);
    dim3 threads(bt);

    tq_encode_paged_kernel<<<grid, threads, smem, stream>>>(
        reinterpret_cast<const half*>(key.data_ptr()),
        reinterpret_cast<const half*>(value.data_ptr()),
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        k_norms.data_ptr<float>(),
        v_norms.data_ptr<float>(),
        cache_seqlens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        pages_per_seq,
        num_kv_heads,
        head_dim,
        k_packed_dim,
        v_packed_dim,
        k_bit_width,
        v_bit_width,
        seq_len,
        k_signs1.data_ptr<float>(),
        k_signs2.data_ptr<float>(),
        v_signs1.data_ptr<float>(),
        v_signs2.data_ptr<float>(),
        k_boundaries.data_ptr<float>(),
        v_boundaries.data_ptr<float>(),
        static_cast<int>(k_boundaries.numel()),
        static_cast<int>(v_boundaries.numel())
    );
    cudaError_t err = cudaPeekAtLastError();
    TORCH_CHECK(err == cudaSuccess, "TurboQuant encode failed: ", cudaGetErrorString(err));
}

void turboquant_decode_paged(
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& k_norms,
    const at::Tensor& v_norms,
    const at::Tensor& k_out,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int k_bit_width,
    int v_bit_width,
    const at::Tensor& k_signs1,
    const at::Tensor& k_signs2,
    const at::Tensor& v_signs1,
    const at::Tensor& v_signs2,
    const at::Tensor& k_centroids,
    const at::Tensor& v_centroids
)
{
    const at::cuda::OptionalCUDAGuard device_guard(key_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(key_cache.dtype() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.dtype() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK_DTYPE(k_out, kHalf);
    TORCH_CHECK_DTYPE(v_out, kHalf);
    TORCH_CHECK(k_out.dim() == 4, "k_out must be 4D");

    int num_kv_heads = k_out.size(2);
    int head_dim = k_out.size(3);
    int k_packed_dim = key_cache.size(3);
    int v_packed_dim = value_cache.size(3);
    int bsz = block_table.size(0);
    int pages_per_seq = block_table.size(1);

    // Upper bound for grid: use pages_per_seq * PAGE_SIZE
    int max_tokens = pages_per_seq * TQ_PAGE_SIZE;

    int bt = ((head_dim + 31) / 32) * 32;
    int smem = head_dim * sizeof(float);

    dim3 grid(max_tokens * num_kv_heads, 1, bsz);
    dim3 threads(bt);

    tq_decode_paged_kernel<<<grid, threads, smem, stream>>>(
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        k_norms.data_ptr<float>(),
        v_norms.data_ptr<float>(),
        reinterpret_cast<half*>(k_out.data_ptr()),
        reinterpret_cast<half*>(v_out.data_ptr()),
        cache_seqlens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(),
        pages_per_seq,
        num_kv_heads,
        head_dim,
        k_packed_dim,
        v_packed_dim,
        k_bit_width,
        v_bit_width,
        k_signs1.data_ptr<float>(),
        k_signs2.data_ptr<float>(),
        v_signs1.data_ptr<float>(),
        v_signs2.data_ptr<float>(),
        k_centroids.data_ptr<float>(),
        v_centroids.data_ptr<float>()
    );
    cudaError_t err = cudaPeekAtLastError();
    TORCH_CHECK(err == cudaSuccess, "TurboQuant decode failed: ", cudaGetErrorString(err));
}

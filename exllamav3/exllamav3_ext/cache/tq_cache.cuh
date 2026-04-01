#pragma once

#include <torch/extension.h>

void turboquant_encode_paged(
    const at::Tensor& key,              // FP16 (num_pages, PAGE_SIZE, num_kv_heads, head_dim)
    const at::Tensor& value,
    const at::Tensor& key_cache,        // uint8 (num_pages, PAGE_SIZE, num_kv_heads, k_packed_dim)
    const at::Tensor& value_cache,
    const at::Tensor& k_norms,          // fp32 (num_pages, PAGE_SIZE, num_kv_heads)
    const at::Tensor& v_norms,
    const at::Tensor& cache_seqlens,    // int32 (batch_size,)
    const at::Tensor& block_table,      // int32 (batch_size, pages_per_seq)
    int seq_len,
    int k_bit_width,
    int v_bit_width,
    const at::Tensor& k_signs1,         // fp32 (head_dim,)
    const at::Tensor& k_signs2,
    const at::Tensor& v_signs1,
    const at::Tensor& v_signs2,
    const at::Tensor& k_boundaries,     // fp32 (num_k_centroids - 1,)
    const at::Tensor& v_boundaries
);

void turboquant_decode_paged(
    const at::Tensor& key_cache,        // uint8 (num_pages, PAGE_SIZE, num_kv_heads, k_packed_dim)
    const at::Tensor& value_cache,
    const at::Tensor& k_norms,          // fp32 (num_pages, PAGE_SIZE, num_kv_heads)
    const at::Tensor& v_norms,
    const at::Tensor& k_out,            // FP16 (num_pages, PAGE_SIZE, num_kv_heads, head_dim)
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,    // int32 (batch_size,)
    const at::Tensor& block_table,      // int32 (batch_size, pages_per_seq)
    int k_bit_width,
    int v_bit_width,
    const at::Tensor& k_signs1,         // fp32 (head_dim,)
    const at::Tensor& k_signs2,
    const at::Tensor& v_signs1,
    const at::Tensor& v_signs2,
    const at::Tensor& k_centroids,      // fp32 (num_k_centroids,)
    const at::Tensor& v_centroids
);

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1
#define MOE_ACT_RELU2_NOGATE 2  // non-gated relu2 (NemotronH): gate GEMM and staging skipped

#define MOE_SMS_PER_EXPERT 8       // default/minimum group width, also sets max concurrency (buffer count)
#define MOE_MAX_SMS_PER_EXPERT 32  // widest expert group when few experts are active
#define MOE_TILESIZE_K 32
// Raising this ALONE makes the kernel slower, which is why it has stood at 16. It is not a
// shape knob: exl3_moe_kernel.cuh passes it as the tile's TILESIZE_M but hardcodes the row
// count three more times at each of the two GEMM call sites -- MIN(size_m, 16) as the rows
// handed to the inner kernel, and `+= 16 * dim` / `size_m -= 16` as the stride between
// strips. Raise the constant and the kernel builds a deeper tile, is still fed 16 rows, and
// still advances 16 rows: TILEBLOCKS_M row blocks of MMA and registers per strip, one of them
// carrying output, and the trellis re-decoded every 16 rows exactly as before. Every row block
// past the filled one is charged in full -- measured at ~17.4 us per 16-row block on a 4090 at
// 5120x17408, ~96% of the tensor roofline for those rows -- so 32 costs about 13% more than 16
// for nothing.
//
// It CAN move, but only with all four numbers together. Worth doing when a census of the
// deployed routing shows experts routinely receiving more than 16 tokens per step; below that
// the deeper tile has no second row block to fill and is pure loss.
#define MOE_TILESIZE_M 16
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#ifndef SMEM_MAX
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6
#endif

#define EXL3_MOE_KERNEL_ARGS                    \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
                                                \
    int* __restrict__ locks

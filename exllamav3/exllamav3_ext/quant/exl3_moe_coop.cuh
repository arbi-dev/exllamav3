#pragma once

#include <ATen/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#define MOE_COOP_THREADS 512
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride; fp16, or bf16 when x_bf16
    bool x_bf16;                // x holds bf16 (the caller's hidden state as it is: no cast, no copy)
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights, fp16 (bf16 when rw_bf16)
    bool rw_bf16;               // rw holds bf16 (the router's own dtype: no cast launch ahead)
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null

    // Range scaling around the fp16 stages (all nullable / off: the unscaled kernels, bit for bit).
    // in_scale: 0-dim fp32 device scalar; the rotated input is divided by it before the fp16
    //   store and the gate/up results multiplied back before the activation (exact: both linear)
    // act_scale: 0-dim fp32 device scalar; the activation is divided by it before the fp16
    //   act_out store and the down results multiplied back in the combine (exact: down is linear)
    // act_auto: per-slot power-of-two activation scale chosen in-kernel from the slot's own peak.
    //   Kernel A stores the activation in fp32 (into gu_u, needs gu_f32); every kernel B block
    //   reads its run's rows, picks scale = 2^ceil(log2(peak / MOE_COOP_ACT_AUTO_PEAK)) (>= 1)
    //   and stages act / scale as fp16; the combine multiplies the slot back by it. Exact, no
    //   host sync, no calibration. Excludes act_scale
    // ovf: int32 device counter (nullable), incremented when a stage leaves the fp16 range or an
    //   output is non-finite. Never reset by the kernels
    const float* in_scale;
    const float* act_scale;
    bool act_auto;
    float* slot_scale;          // (slots_max,) fp32 scratch, act_auto only
    int* ovf;
};

// act_auto: the stored activation peak bound. Leaves the down stage's fp16 accumulation
// (64 products before each fp32 fold) headroom for the trellis weights' magnitude
#define MOE_COOP_ACT_AUTO_PEAK 1024.0f

// Launch with plain device pointers (called from BC_BlockSparseMLP). K: gate/up and down bit
// widths (gate and up always share one, the converter allocates them as one group); cb: 0 default,
// 1 mcg, 2 mul1 codebook, uniform across the three projections
void exl3_moe_coop_launch(const MoeCoopParams& p, int K_gu, int K_d, int cb, int device, cudaStream_t stream);

// Static part of the parameter block, validated once from the module's tensors (BC construction);
// K_gu / K_d / cb come back through the out-params. Gate tables/scratch are ignored when
// gated == false but must still be valid tensors (pass the up ones)
MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    int& K_gu, int& K_d, int& cb
);

// Per-call part: input rows, routing, optional shared-expert output (bsz rows, width H), launch
void exl3_moe_coop_run
(
    MoeCoopParams p, int K_gu, int K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
);

// Range-scaled variant of exl3_moe_coop_run (see MoeCoopParams: in_scale / act_scale / act_auto)
void exl3_moe_coop_run_scaled
(
    MoeCoopParams p, int K_gu, int K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& in_scale,
    const c10::optional<at::Tensor>& act_scale,
    const c10::optional<at::Tensor>& slot_scale,
    const c10::optional<at::Tensor>& ovf
);

// Tensor front end (prepare + run; the test entry point)
void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
);

// Length of the int32 scratch for a module: completion counters + run table (slots_max =
// MAX_BSZN * top_k)
inline int64_t exl3_moe_coop_ctr_len(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max;
}

// Tensor front end with range scaling. slot_scale (fp32, >= slots of the scratch) given means
// act_auto (excludes act_scale); in_scale / act_scale are 0-dim fp32 device tensors; ovf a 1-elem
// int32 device counter. All four nullable: with none this is exl3_moe_coop exactly
void exl3_moe_coop_scaled
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w,
    const c10::optional<at::Tensor>& in_scale,
    const c10::optional<at::Tensor>& act_scale,
    const c10::optional<at::Tensor>& slot_scale,
    const c10::optional<at::Tensor>& ovf
);

// What the loaded binary's coop entries accept beyond the first release, as a bit set read by the
// caller instead of a version: 1 = bf16 routing weights, 2 = bf16 input rows (exl3_moe_coop*, exl3_moe_coop2)
#define EXL3_MOE_COOP_CAP_RW_BF16 1
#define EXL3_MOE_COOP_CAP_X_BF16 2
inline int64_t exl3_moe_coop_caps() { return EXL3_MOE_COOP_CAP_RW_BF16 | EXL3_MOE_COOP_CAP_X_BF16; }

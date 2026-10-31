#pragma once

#include <ATen/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// exl3_moe_coop2: the routed mixture of a decode / verify step (1 .. MOE_COOP2_MAX_SLOTS / top_k
// rows) in two launches, every distinct expert's weights streamed once per step.
//
//   A (gate/up): units (run, projection, 256-column group, k segment). A run is up to 16 slots
//                that picked one expert: the rows of one m16n8k16 MMA. The unit rotates its rows'
//                input over its k segment itself (x * suh, 128-wide Hadamard) and streams its
//                expert's trellis through a per-warp cp.async ring. Partial sums go to scratch;
//                the last arrival per (run, group) sums the k segments in a fixed order and runs
//                the epilogue (output rotation * svh, activation, * down suh, rotation), storing
//                the fp32 activation and its per-(slot, 128-chunk) peak.
//   B (down):    units (run, 256-column group, k segment) on the activation, staged as fp16 at a
//                per-slot power-of-two scale chosen from the slot's peak (act_auto). The last
//                arrival per (token row, group) sums each slot's k segments, rotates * svh,
//                rescales, weights and sums the token's slots in a fixed order, adds the shared
//                expert term and stores the row.
//
// Both launches size their grid to the card (persistent blocks looping over units) and split k
// so a 1-row step still fills every SM. Every block builds the step's run table itself from
// the routing (no pre-kernel). Deterministic: fixed-order reductions, and a row's result does
// not depend on which other rows share its run.

#define MOE_COOP2_ACT_SILU 0
#define MOE_COOP2_ACT_GELU 1
#define MOE_COOP2_ACT_RELU2 2

#define MOE_COOP2_MAX_SLOTS 512
#define MOE_COOP2_MAX_EXPERTS 1024
#define MOE_COOP2_GROUP_COLS 256

// act_auto: the stored activation peak bound (the down stage accumulates in fp32; the bound
// keeps the fp16 operand itself well inside range)
#define MOE_COOP2_ACT_AUTO_PEAK 1024.0f

struct MoeCoop2Params
{
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert ids
    const half* rw;             // (bsz, topk) routing weights; 0 = inactive slot
    int bsz;
    int topk;
    int slots;                  // bsz * topk
    int H;                      // x width (<= Hi)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: unsharded; else ids in [min, max) are local
    int max_expert;
    int n_local;

    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    bool gated;
    int act;

    float* part_a;              // (nproj, ks_a, slots, I) gate/up partial sums
    float* act32;               // (slots_cap, I) activation, fp32
    float* peak;                // (slots_cap, I / 128) activation peak per 128-chunk
    float* part_b;              // (ks_b, slots, Ho) down partial sums
    int* ctr_a;                 // (slots_cap, cg_a) arrivals per (run, group)
    int* ctr_b;                 // (slots_cap, cg_b) arrivals per (token row, group)
    int ctr_a_len;
    int ctr_b_len;
    int cg_a;                   // 256-column groups of I
    int cg_b;                   // 256-column groups of Ho
    int ks_a;                   // k segments per gate/up unit
    int ks_b;                   // k segments per down unit
    bool act_auto;

    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;
    const float* sh_out;        // (bsz, H) shared-expert output added to every row, nullable
    int* ovf;                   // fp16-range overflows / non-finite values, nullable, never reset
    int pfd;                    // L2 prefetch distance in k slices (EXL3_MOE_COOP2_PFD)
    int dbg;                    // EXL3_MOE_COOP2_DBG (profiling): 1 skip dequant+MMA, 2 skip operand staging,
                                // 4 no L2 prefetch, 8 no evict-first hint
};

// Scratch lengths for a geometry (slots_cap = rows_max * top_k <= MOE_COOP2_MAX_SLOTS)
inline int64_t exl3_moe_coop2_ctr_len(int64_t slots_cap, int64_t I, int64_t Ho)
{
    const int64_t g = MOE_COOP2_GROUP_COLS;
    return slots_cap * ((I + g - 1) / g) + slots_cap * ((Ho + g - 1) / g);
}

void exl3_moe_coop2
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
    int Kg, int Ku, int Kd,
    bool mcg, bool mul1,
    int act,
    bool gated,
    at::Tensor& part_a,
    at::Tensor& act32,
    at::Tensor& peak,
    at::Tensor& part_b,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& ovf,
    bool act_auto,
    int ks_a,
    int ks_b
);

// The launch plan exl3_moe_coop2 would use for this step: {ks_a, ks_b, grid_a, grid_b}
std::vector<int64_t> exl3_moe_coop2_plan
(
    int bsz, int topk, int Hi, int I, int Ho, bool gated, int K_gu, int K_d, int slots_cap, int device,
    int ks_a, int ks_b
);

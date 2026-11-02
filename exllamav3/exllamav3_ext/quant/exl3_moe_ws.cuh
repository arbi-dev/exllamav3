#pragma once

#include <ATen/Tensor.h>
#include <vector>

// exl3_moe_coop2, warp-specialised mode (exl3_moe_ws_kernel.cuh): exl3_moe_coop2's arguments and
// scratch (the counter scratch at its full length), then the run length (16 or 32 rows, 0: by the
// step's rows) and the grid fill in percent of the resident blocks (0: EXL3_MOE_COOP2_WS_FILL or 100)
void exl3_moe_coop2_ws
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
    int ks_b,
    int rm,
    int fill_pct,
    const c10::optional<at::Tensor>& s_ptrs,     // the shared expert folded in: [gate, up, down] x [trellis, suh, svh] (ungated: gate = up)
    int K_s,                                     // its width
    int I_s,                                     // its intermediate width
    const c10::optional<at::Tensor>& s_scratch,  // fp32, exl3_moe_coop2_ws_shared_scratch()[0] elements
    const c10::optional<at::Tensor>& s_ctr,      // int32, [1] elements, zero before the first call
    int s_first                                  // 1: shared units ahead of the routed ones
);

// Shared-expert scratch for steps of up to rows_max rows: {fp32 elements, int32 elements}
std::vector<int64_t> exl3_moe_coop2_ws_shared_scratch(int rows_max, int I_s, int Ho, bool gated);

// Profiling (EXL3_MOE_COOP2_TRACE=1): the last traced call's stamps and kernel A's grid. Per block:
// [0] start, [1] run table built, [2] exit, [3] units | ks << 32 | ks_shared << 48; per unit i < 20:
// [4 + 3i] start, [5 + 3i] consumed, [6 + 3i] arrival and epilogue / combine done
std::vector<at::Tensor> exl3_moe_coop2_ws_trace();

// Whether a K_s-bit shared expert folds into K_gu / K_d-bit routed experts
bool exl3_moe_coop2_ws_can_fold(int K_gu, int K_d, int K_s);

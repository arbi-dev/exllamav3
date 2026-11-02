#pragma once

#include <ATen/Tensor.h>

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
    int fill_pct
);

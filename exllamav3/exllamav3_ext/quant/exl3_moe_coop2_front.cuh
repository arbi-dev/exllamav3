#pragma once

// exl3_moe_coop2 front end shared by the coop2 and ws launchers: argument checks and the step's
// MoeCoop2Params (everything but the k split, the scratch counters' layout and the launch)

#include <ATen/ATen.h>
#include <cstdlib>
#include "../util.h"
#include "exl3_moe_coop2.cuh"

inline int coop2_env(const char* name, int dflt)
{
    const char* v = std::getenv(name);
    return v ? atoi(v) : dflt;
}

inline MoeCoop2Params coop2_front
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
    bool act_auto,
    int kc,
    int maxc_b,
    int& slots_cap
)
{
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "exl3_moe_coop2: x must be fp16 or bf16");
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK(rw.scalar_type() == at::kHalf || rw.scalar_type() == at::kBFloat16, "exl3_moe_coop2: rw must be fp16 or bf16");
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 4 == 0, "exl3_moe_coop2: x must be (bsz, H), unit column stride, row stride a multiple of 4");
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous() && sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0),
                "exl3_moe_coop2: sel / rw must be contiguous (bsz, top_k)");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop2: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop2: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop2: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop2: gate and up must share a bit width");
    TORCH_CHECK(act == MOE_COOP2_ACT_SILU || act == MOE_COOP2_ACT_GELU || act == MOE_COOP2_ACT_RELU2, "exl3_moe_coop2: unknown activation");
    for (const at::Tensor* t : { &part_a, &act32, &peak, &part_b, &out })
        TORCH_CHECK(t->scalar_type() == at::kFloat, "exl3_moe_coop2: fp32 scratch / output");
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(part_a.is_contiguous() && act32.is_contiguous() && peak.is_contiguous() && part_b.is_contiguous() && ctr.is_contiguous(),
                "exl3_moe_coop2: scratch must be contiguous");

    MoeCoop2Params p = {};
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.x_bf16 = x.scalar_type() == at::kBFloat16;
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    p.rw_bf16 = rw.scalar_type() == at::kBFloat16;
    p.bsz = (int) x.size(0);
    p.topk = (int) sel.size(1);
    p.slots = p.bsz * p.topk;
    p.H = (int) x.size(1);
    p.Hi = Hi;
    p.I = (int) act32.size(-1);
    p.Ho = (int) part_b.size(-1);
    p.H_out = (int) out.size(-1);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    p.n_local = (int) u_trellis.size(0);
    TORCH_CHECK(p.Hi % kc == 0 && p.I % kc == 0 && p.Ho % kc == 0, "exl3_moe_coop2: Hi / I / Ho must be multiples of 128");
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi && p.H_out <= p.Ho, "exl3_moe_coop2: H / H_out shape");
    TORCH_CHECK(p.n_local <= MOE_COOP2_MAX_EXPERTS, "exl3_moe_coop2: more than ", MOE_COOP2_MAX_EXPERTS, " local experts");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop2: expert range exceeds tables");
    TORCH_CHECK(out.dim() == 2 && out.stride(-1) == 1 && out.size(0) >= p.bsz && out.stride(0) % 4 == 0, "exl3_moe_coop2: out shape");

    slots_cap = (int) act32.size(0);
    TORCH_CHECK(slots_cap <= MOE_COOP2_MAX_SLOTS && p.slots >= 1 && p.slots <= slots_cap, "exl3_moe_coop2: step slots exceed the scratch");
    TORCH_CHECK(peak.numel() >= (int64_t) slots_cap * (p.I / kc), "exl3_moe_coop2: peak scratch too small");
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop2_ctr_len_2l(slots_cap, p.I, p.Ho), "exl3_moe_coop2: counter scratch too small");

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.gated = gated;
    p.act = act;
    p.act_auto = act_auto;
    p.dbg = coop2_env("EXL3_MOE_COOP2_DBG", 0);
    p.pfd = coop2_env("EXL3_MOE_COOP2_PFD", 0);
    p.pfd_b = coop2_env("EXL3_MOE_COOP2_PFD_B", 0);
    p.maxc_b = maxc_b;
    return p;
}

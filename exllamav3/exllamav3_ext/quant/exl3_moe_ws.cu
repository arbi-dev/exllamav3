// exl3_moe_coop2, warp-specialised mode: launcher. Kernels in exl3_moe_ws_kernel.cuh, instantiated
// per K in comp_units/exl3_moe_ws_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop2.cuh"
#include "exl3_moe_coop2_front.cuh"
#include "exl3_moe_ws.cuh"
#include "comp_units/exl3_moe_ws_instances.cuh"
#include "exl3_moe_ws_kernel.cuh"

namespace {

#ifndef EXL3_MOE_WS_KSET
#define EXL3_MOE_WS_KSET(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8)
#endif

void* ws_kernel(bool is_b, int K, int Ks, int cb, int rm)
{
    switch (K)
    {
        #define EXL3_MOE_WS_CASE(k) case k: return is_b ? exl3_moe_ws_kernel_b_k##k(Ks, cb, rm) : exl3_moe_ws_kernel_a_k##k(Ks, cb, rm);
        EXL3_MOE_WS_KSET(EXL3_MOE_WS_CASE)
        #undef EXL3_MOE_WS_CASE
        default: return nullptr;
    }
}

int ws_smem(int K, int rm)
{
    using namespace exl3_moe_ws_ns;
    switch (K * 64 + rm)
    {
        #define EXL3_MOE_WS_SM(k) case k * 64 + 16: return smem_bytes<k, 16>(); case k * 64 + 32: return smem_bytes<k, 32>();
        EXL3_MOE_WS_KSET(EXL3_MOE_WS_SM)
        #undef EXL3_MOE_WS_SM
        default: return 0;
    }
}

// Resident blocks per SM (once per kernel and device), the dynamic smem limit raised first
int ws_occupancy(void* kernel, int smem, int device)
{
    static std::mutex mtx;
    static std::map<std::pair<void*, int>, int> cache;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(kernel, device);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int n = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, exl3_moe_ws_ns::WTHREADS, smem));
    n = std::max(n, 1);
    cache[key] = n;
    return n;
}

int env_int_ws(const char* name, int dflt)
{
    const char* v = std::getenv(name);
    return v ? atoi(v) : dflt;
}

// Shared-expert scratch of a step of `rows` rows: fp32 [part_a_s | act32_s | peak_s | part_b_s |
// slot_w_s] and int32 [ctr_a_s], the k split capped at WS_CAP_S segments
constexpr int WS_CAP_S = 8;

struct WsSharedLayout { int64_t part_a, act32, peak, part_b, slot_w, f32, ctr; };

WsSharedLayout ws_shared_layout(int64_t rows, int64_t I_s, int64_t Ho, bool gated)
{
    WsSharedLayout l;
    const int64_t nproj = gated ? 2 : 1;
    l.part_a = 0;
    l.act32 = l.part_a + nproj * WS_CAP_S * rows * I_s;
    l.peak = l.act32 + rows * I_s;
    l.part_b = l.peak + rows * (I_s / 128);
    l.slot_w = l.part_b + WS_CAP_S * rows * Ho;
    l.f32 = l.slot_w + rows;
    l.ctr = ((rows + 15) / 16) * ((I_s + 255) / 256);
    return l;
}

}  // namespace

std::vector<int64_t> exl3_moe_coop2_ws_shared_scratch(int rows_max, int I_s, int Ho, bool gated)
{
    const WsSharedLayout l = ws_shared_layout(rows_max, I_s, Ho, gated);
    return { l.f32, l.ctr };
}

bool exl3_moe_coop2_ws_can_fold(int K_gu, int K_d, int K_s)
{
    return (K_s == K_gu || K_s == 6) && (K_s == K_d || K_s == 6);
}

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
    const c10::optional<at::Tensor>& s_ptrs,
    int K_s,
    int I_s,
    const c10::optional<at::Tensor>& s_scratch,
    const c10::optional<at::Tensor>& s_ctr,
    int s_first
)
{
    using namespace exl3_moe_ws_ns;
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    int slots_cap = 0;
    MoeCoop2Params p = coop2_front(x, sel, rw, min_expert, max_expert, Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh,
                                   d_trellis, d_suh, d_svh, Kg, Ku, Kd, mcg, mul1, act, gated, part_a, act32, peak, part_b,
                                   ctr, out, act_auto, KC, env_int_ws("EXL3_MOE_COOP2_MAXC_B", MOE_COOP2_MAX_DOWN_CHUNKS), slots_cap);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop2_ctr_len(slots_cap, p.I, p.Ho), "exl3_moe_coop2_ws: counter scratch too small");

    const int K_gu = gated ? Kg : Ku;
    const int cb = mcg ? 1 : (mul1 ? 2 : 0);
    if (rm <= 0) rm = p.bsz <= 16 ? 16 : 32;
    TORCH_CHECK(rm == 16 || rm == 32, "exl3_moe_coop2_ws: run length must be 16 or 32");
    const bool fold = s_ptrs.has_value();
    if (fold)
    {
        TORCH_CHECK(exl3_moe_coop2_ws_can_fold(K_gu, Kd, K_s), "exl3_moe_coop2_ws: no kernel folds a ", K_s, "-bit shared expert into ", K_gu, "/", Kd, "-bit experts");
        TORCH_CHECK(!sh_out, "exl3_moe_coop2_ws: a folded shared expert takes no shared output");
        TORCH_CHECK(I_s > 0 && I_s % KC == 0, "exl3_moe_coop2_ws: shared intermediate width must be a multiple of 128");
        TORCH_CHECK(s_ptrs->scalar_type() == at::kLong && s_ptrs->numel() == 9 && s_ptrs->is_contiguous(), "exl3_moe_coop2_ws: s_ptrs must be 9 contiguous int64");
        TORCH_CHECK(s_scratch && s_ctr, "exl3_moe_coop2_ws: a folded shared expert needs its scratch");
        const WsSharedLayout l = ws_shared_layout(p.bsz, I_s, p.Ho, gated);
        TORCH_CHECK(s_scratch->scalar_type() == at::kFloat && s_scratch->is_contiguous() && s_scratch->numel() >= l.f32, "exl3_moe_coop2_ws: shared scratch too small");
        TORCH_CHECK(s_ctr->scalar_type() == at::kInt && s_ctr->is_contiguous() && s_ctr->numel() >= l.ctr, "exl3_moe_coop2_ws: shared counter scratch too small");
        float* f = (float*) s_scratch->data_ptr();
        p.s_ptrs = (const int64_t*) s_ptrs->data_ptr();
        p.I_s = I_s;
        p.cg_as = CEIL_DIVIDE(I_s, GCOLS);
        p.cap_s = WS_CAP_S;
        p.part_a_s = f + l.part_a;
        p.act32_s = f + l.act32;
        p.peak_s = f + l.peak;
        p.part_b_s = f + l.part_b;
        p.slot_w_s = f + l.slot_w;
        p.ctr_a_s = (int*) s_ctr->data_ptr();
        p.ctr_a_s_len = (int) l.ctr;
        p.s_first = s_first;
    }
    const int Ks_a = fold ? K_s : K_gu, Ks_b = fold ? K_s : Kd;
    void* ka = ws_kernel(false, K_gu, Ks_a, cb, rm);
    void* kb = ws_kernel(true, Kd, Ks_b, cb, rm);
    TORCH_CHECK(ka && kb, "exl3_moe_coop2_ws: no kernel for this width");
    const int sm_a = ws_smem(std::max(K_gu, Ks_a), rm), sm_b = ws_smem(std::max(Kd, Ks_b), rm);

    const int nproj = gated ? 2 : 1;
    p.cap_a = (int) std::min<int64_t>(part_a.numel() / ((int64_t) nproj * p.I), 1 << 20);
    p.cap_b = (int) std::min<int64_t>(part_b.numel() / p.Ho, 1 << 20);
    TORCH_CHECK(p.cap_a >= p.slots && p.cap_b >= p.slots, "exl3_moe_coop2_ws: partial-sum scratch smaller than the step's slots");
    p.ks_a = ks_a > 0 ? std::min(ks_a, std::min(p.Hi / KC, p.cap_a / p.slots)) : 0;
    p.ks_b = ks_b > 0 ? std::min(ks_b, std::min(p.I / KC, p.cap_b / p.slots)) : 0;
    p.cg_a = CEIL_DIVIDE(p.I, GCOLS);
    p.cg_b = CEIL_DIVIDE(p.Ho, GCOLS);

    p.part_a = (float*) part_a.data_ptr();
    p.act32 = (float*) act32.data_ptr();
    p.peak = (float*) peak.data_ptr();
    p.part_b = (float*) part_b.data_ptr();
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_a_len = slots_cap * p.cg_a;
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.ctr_b_len = slots_cap * p.cg_b;
    p.slot_w = (float*) (p.ctr_b + p.ctr_b_len + 2 + slots_cap);
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(0);

    if (sh_out)
    {
        TORCH_CHECK(sh_out->scalar_type() == at::kFloat && sh_out->is_contiguous() && sh_out->size(-1) == p.H &&
                    sh_out->numel() >= (int64_t) p.bsz * p.H, "exl3_moe_coop2_ws: shared expert output must be contiguous fp32 (bsz, H)");
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop2_ws: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
    }
    if (ovf)
    {
        TORCH_CHECK(ovf->scalar_type() == at::kInt && ovf->numel() >= 1 && ovf->device() == x.device(), "exl3_moe_coop2_ws: ovf must be an int32 device counter");
        p.ovf = (int*) ovf->data_ptr();
    }

    // Persistent grids sized to the card, at most one block per possible unit
    if (fill_pct <= 0) fill_pct = env_int_ws("EXL3_MOE_COOP2_WS_FILL", 100);
    const int sms = DevCtx::instance().get_num_sms(device);
    const int ca = p.Hi / KC, cbk = p.I / KC;
    int max_a = p.slots * nproj * p.cg_a * std::max(1, std::min(ca, p.cap_a / p.slots));
    int max_b = p.slots * p.cg_b * std::max(1, std::min(cbk, p.cap_b / p.slots));
    if (fold)
    {
        const int n_sr = CEIL_DIVIDE(p.bsz, 16);
        max_a += n_sr * nproj * p.cg_as * std::min(ca, WS_CAP_S);
        max_b += n_sr * p.cg_b * std::min(I_s / KC, WS_CAP_S);
    }
    const int grid_a = std::min(max_a, std::max(1, sms * ws_occupancy(ka, sm_a, device) * fill_pct / 100));
    const int grid_b = std::min(max_b, std::max(1, sms * ws_occupancy(kb, sm_b, device) * fill_pct / 100));

    void* args[] = { (void*) &p };
    // EXL3_MOE_COOP2_WS_ONLY (profiling): 1 kernel A alone, 2 kernel B alone (on the last step's activation)
    const int only = env_int_ws("EXL3_MOE_COOP2_WS_ONLY", 0);
    if (only != 2) cuda_check(cudaLaunchKernel(ka, dim3(grid_a), dim3(WTHREADS), args, sm_a, stream));
    if (only != 1) cuda_check(cudaLaunchKernel(kb, dim3(grid_b), dim3(WTHREADS), args, sm_b, stream));
}

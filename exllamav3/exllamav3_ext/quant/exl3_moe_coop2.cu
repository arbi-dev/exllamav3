// exl3_moe_coop2: launcher and tensor front end. Kernels in exl3_moe_coop2_kernel.cuh,
// instantiated per (K, codebook) in comp_units/exl3_moe_coop2_inst_k*.cu

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
#include "comp_units/exl3_moe_coop2_instances.cuh"
#include "exl3_moe_coop2_kernel.cuh"

namespace {

void* coop2_kernel_a(int K, int cb)
{
    switch (K)
    {
        case 1: return exl3_moe_coop2_kernel_a_k1(cb);
        case 2: return exl3_moe_coop2_kernel_a_k2(cb);
        case 3: return exl3_moe_coop2_kernel_a_k3(cb);
        case 4: return exl3_moe_coop2_kernel_a_k4(cb);
        case 5: return exl3_moe_coop2_kernel_a_k5(cb);
        case 6: return exl3_moe_coop2_kernel_a_k6(cb);
        case 7: return exl3_moe_coop2_kernel_a_k7(cb);
        default: return exl3_moe_coop2_kernel_a_k8(cb);
    }
}

void* coop2_kernel_b(int K, int cb)
{
    switch (K)
    {
        case 1: return exl3_moe_coop2_kernel_b_k1(cb);
        case 2: return exl3_moe_coop2_kernel_b_k2(cb);
        case 3: return exl3_moe_coop2_kernel_b_k3(cb);
        case 4: return exl3_moe_coop2_kernel_b_k4(cb);
        case 5: return exl3_moe_coop2_kernel_b_k5(cb);
        case 6: return exl3_moe_coop2_kernel_b_k6(cb);
        case 7: return exl3_moe_coop2_kernel_b_k7(cb);
        default: return exl3_moe_coop2_kernel_b_k8(cb);
    }
}

void* coop2_kernel_f(int K, int cb)
{
    switch (K)
    {
        case 1: return exl3_moe_coop2_kernel_f_k1(cb);
        case 2: return exl3_moe_coop2_kernel_f_k2(cb);
        case 3: return exl3_moe_coop2_kernel_f_k3(cb);
        case 4: return exl3_moe_coop2_kernel_f_k4(cb);
        case 5: return exl3_moe_coop2_kernel_f_k5(cb);
        case 6: return exl3_moe_coop2_kernel_f_k6(cb);
        case 7: return exl3_moe_coop2_kernel_f_k7(cb);
        default: return exl3_moe_coop2_kernel_f_k8(cb);
    }
}

// Resident blocks per SM of a kernel (once per kernel and device)
int coop2_occupancy(void* kernel, int device)
{
    static std::mutex mtx;
    static std::map<std::pair<void*, int>, int> cache;
    std::lock_guard<std::mutex> lock(mtx);
    auto key = std::make_pair(kernel, device);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    int n = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, exl3_moe_coop2_ns::THREADS, exl3_moe_coop2_ns::SMEM_BYTES));
    n = std::max(n, 1);
    cache[key] = n;
    return n;
}

int env_int(const char* name, int dflt)
{
    const char* v = std::getenv(name);
    return v ? atoi(v) : dflt;
}

struct Coop2Plan { int ks_a, ks_b, grid_a, grid_b; };

// Persistent grids sized to the card (EXL3_MOE_COOP2_FILL scales them, racing), at most one block
// per possible unit. The k split is the kernels' own per-step choice (pick_ks, from the step's
// run count) unless forced (ks_a / ks_b, or EXL3_MOE_COOP2_KSA / _KSB); the plan reports the
// split a step whose slots all pick distinct experts would get (exact for one row).
Coop2Plan coop2_plan(int bsz, int topk, int Hi, int I, int Ho, bool gated, int cap_a, int cap_b, int device,
                     void* ka, void* kb, int ks_a_req, int ks_b_req, bool resolve)
{
    using namespace exl3_moe_coop2_ns;
    const int slots = bsz * topk;
    const int nproj = gated ? 2 : 1;
    const int cg_a = CEIL_DIVIDE(I, GCOLS);
    const int cg_b = CEIL_DIVIDE(Ho, GCOLS);
    const int sms = DevCtx::instance().get_num_sms(device);
    const int fill_pct = env_int("EXL3_MOE_COOP2_FILL", 100);
    const int fill_a = std::max(1, sms * coop2_occupancy(ka, device) * fill_pct / 100);
    const int fill_b = std::max(1, sms * coop2_occupancy(kb, device) * fill_pct / 100);
    const int ca = Hi / KC, cbk = I / KC;
    const int kcap_a = std::max(1, std::min(ca, cap_a / slots));
    const int kcap_b = std::max(1, std::min(cbk, cap_b / slots));
    Coop2Plan pl;
    pl.grid_a = std::min(fill_a, slots * nproj * cg_a * kcap_a);
    pl.grid_b = std::min(fill_b, slots * cg_b * kcap_b);
    const int fa = ks_a_req > 0 ? ks_a_req : env_int("EXL3_MOE_COOP2_KSA", 0);
    const int fb = ks_b_req > 0 ? ks_b_req : env_int("EXL3_MOE_COOP2_KSB", 0);
    pl.ks_a = fa > 0 ? std::min(fa, kcap_a) : (resolve ? exl3_moe_coop2_ns::pick_ks(slots * nproj * cg_a, ca, pl.grid_a, cap_a / slots) : 0);
    pl.ks_b = fb > 0 ? std::min(fb, kcap_b) : (resolve ? exl3_moe_coop2_ns::pick_ks(slots * cg_b, cbk, pl.grid_b, cap_b / slots) : 0);
    return pl;
}

int coop2_cb(bool mcg, bool mul1) { return mcg ? 1 : (mul1 ? 2 : 0); }

at::Tensor g_trace;

}  // namespace

at::Tensor exl3_moe_coop2_trace() { return g_trace.defined() ? g_trace.cpu() : at::Tensor(); }

std::vector<int64_t> exl3_moe_coop2_plan
(
    int bsz, int topk, int Hi, int I, int Ho, bool gated, int K_gu, int K_d, int cap_a, int cap_b, int device,
    int ks_a, int ks_b
)
{
    const Coop2Plan pl = coop2_plan(bsz, topk, Hi, I, Ho, gated, cap_a, cap_b, device,
                                    coop2_kernel_a(K_gu, 0), coop2_kernel_b(K_d, 0), ks_a, ks_b, true);
    return { pl.ks_a, pl.ks_b, pl.grid_a, pl.grid_b };
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
)
{
    using namespace exl3_moe_coop2_ns;
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    TORCH_CHECK_DTYPE(x, kHalf);
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
    TORCH_CHECK(p.Hi % KC == 0 && p.I % KC == 0 && p.Ho % KC == 0, "exl3_moe_coop2: Hi / I / Ho must be multiples of 128");
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi && p.H_out <= p.Ho, "exl3_moe_coop2: H / H_out shape");
    TORCH_CHECK(p.n_local <= MOE_COOP2_MAX_EXPERTS, "exl3_moe_coop2: more than ", MOE_COOP2_MAX_EXPERTS, " local experts");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop2: expert range exceeds tables");
    TORCH_CHECK(out.dim() == 2 && out.stride(-1) == 1 && out.size(0) >= p.bsz && out.stride(0) % 4 == 0, "exl3_moe_coop2: out shape");

    const int slots_cap = (int) act32.size(0);
    TORCH_CHECK(slots_cap <= MOE_COOP2_MAX_SLOTS && p.slots >= 1 && p.slots <= slots_cap, "exl3_moe_coop2: step slots exceed the scratch");
    TORCH_CHECK(peak.numel() >= (int64_t) slots_cap * (p.I / KC), "exl3_moe_coop2: peak scratch too small");
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
    p.dbg = env_int("EXL3_MOE_COOP2_DBG", 0);
    p.pfd = env_int("EXL3_MOE_COOP2_PFD", 0);
    p.pfd_b = env_int("EXL3_MOE_COOP2_PFD_B", 0);

    const int K_gu = gated ? Kg : Ku;
    const int cb = coop2_cb(mcg, mul1);
    void* ka = coop2_kernel_a(K_gu, cb);
    void* kb = coop2_kernel_b(Kd, cb);
    const int nproj = gated ? 2 : 1;
    p.cap_a = (int) std::min<int64_t>(part_a.numel() / ((int64_t) nproj * p.I), 1 << 20);
    p.cap_b = (int) std::min<int64_t>(part_b.numel() / p.Ho, 1 << 20);
    TORCH_CHECK(p.cap_a >= p.slots && p.cap_b >= p.slots, "exl3_moe_coop2: partial-sum scratch smaller than the step's slots");
    const Coop2Plan pl = coop2_plan(p.bsz, p.topk, p.Hi, p.I, p.Ho, gated, p.cap_a, p.cap_b, device, ka, kb, ks_a, ks_b, false);
    p.ks_a = pl.ks_a;
    p.ks_b = pl.ks_b;
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
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(0);

    if (sh_out)
    {
        TORCH_CHECK(sh_out->scalar_type() == at::kFloat && sh_out->is_contiguous() && sh_out->size(-1) == p.H &&
                    sh_out->numel() >= (int64_t) p.bsz * p.H, "exl3_moe_coop2: shared expert output must be contiguous fp32 (bsz, H)");
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop2: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
    }
    if (ovf)
    {
        TORCH_CHECK(ovf->scalar_type() == at::kInt && ovf->numel() >= 1 && ovf->device() == x.device(), "exl3_moe_coop2: ovf must be an int32 device counter");
        p.ovf = (int*) ovf->data_ptr();
    }

    void* args[] = { (void*) &p };

    // One launch when gate/up and down share a width and the counter scratch holds the fused
    // kernel's queue (EXL3_MOE_COOP2_FUSED=0: the two launches, comparison arm)
    const bool full_ctr = ctr.numel() >= exl3_moe_coop2_ctr_len(slots_cap, p.I, p.Ho);
    if (full_ctr && !(p.dbg & 64)) p.slot_w = (float*) (p.ctr_b + p.ctr_b_len + 2 + slots_cap);
    const bool fused = K_gu == Kd && env_int("EXL3_MOE_COOP2_FUSED", 1) != 0 && full_ctr;
    if (fused)
    {
        void* kf = coop2_kernel_f(K_gu, cb);
        p.fctr = p.ctr_b + p.ctr_b_len;
        // 80% of the resident slots: the queue keeps the card's DRAM busy with fewer blocks, and
        // the rest stay free for the kernels a serving step runs beside the mixture on another
        // stream (the shared expert under moe_shared_overlap), which a full grid would serialise
        const int fill_pct = env_int("EXL3_MOE_COOP2_FILL", 80);
        const int fill = std::max(1, DevCtx::instance().get_num_sms(device) * coop2_occupancy(kf, device) * fill_pct / 100);
        const int ca = p.Hi / KC, cbk = p.I / KC;
        const int max_units = p.slots * nproj * p.cg_a * std::max(1, std::min(ca, p.cap_a / p.slots)) +
                              p.slots * p.cg_b * std::max(1, std::min(cbk, p.cap_b / p.slots));
        const int grid = std::min(fill, max_units);
        if (env_int("EXL3_MOE_COOP2_TRACE", 0))
        {
            g_trace = at::zeros({grid, exl3_moe_coop2_ns::TRACE_W}, x.options().dtype(at::kLong));
            p.trace = (long long*) g_trace.data_ptr();
        }
        cuda_check(cudaLaunchKernel(kf, dim3(grid), dim3(THREADS), args, SMEM_BYTES, stream));
        return;
    }
    cuda_check(cudaLaunchKernel(ka, dim3(pl.grid_a), dim3(THREADS), args, SMEM_BYTES, stream));
    cuda_check(cudaLaunchKernel(kb, dim3(pl.grid_b), dim3(THREADS), args, SMEM_BYTES, stream));
}

#pragma once

// exl3_moe_coop2, warp-specialised mode (ws): the routed mixture as two launches, gate/up (A)
// then down (B), each a single unit kind with coop2's unit decomposition and reduction order (so
// a row's result is bitwise coop2's at the same k split). Per block:
//
//   consumer warps (NCW): decode the trellis and MMA, 32 columns each; accumulators live only here
//   weight producer (1 warp): a block-shared NSTG-stage cp.async ring, signalled through mbarriers
//                    (cp.async.mbarrier.arrive.noinc); it runs ahead across unit edges
//   operand stagers (NSW warps): the unit's rows as fp16 MMA operands per 128-k chunk (A: x * suh,
//                    Hadamard; B: activation * inverse slot scale) into NOB buffers
//
// No block barrier per chunk: stages hand over through full/empty mbarriers, the consumer warps
// meet on a named barrier only at a unit's arrival (one thread releases, gpu scope). Ring cursors
// are registers; the ring and the operand buffers are powers of two.

#include "exl3_moe_coop2_kernel.cuh"

namespace exl3_moe_ws_ns {

using namespace exl3_moe_coop2_ns;

constexpr int NCW = NW;                         // consumer warps, WNT 16-column tiles each
constexpr int WPW = NCW;                        // weight producer warp
constexpr int SPW = NCW + 1;                    // first operand stager warp
constexpr int NSW = 2;                          // operand stager warps (rows split between them)
constexpr int NWS = NCW + 1 + NSW;
constexpr int WTHREADS = NWS * 32;
constexpr int CTHREADS = NCW * 32;
constexpr int NSTG = 4;                         // weight ring stages
constexpr int SPS = 2;                          // 16-k slices per stage
constexpr int STG_PER_CHUNK = SPC / SPS;
constexpr int NOB = 2;                          // operand buffers, one 128-k chunk each
constexpr int CBAR = 1;                         // named barrier: the consumer warps

static_assert((NSTG & (NSTG - 1)) == 0 && (NOB & (NOB - 1)) == 0, "ring sizes must be powers of two");
static_assert(SPC % SPS == 0, "stage must divide the chunk");

template <int bits> __host__ __device__ constexpr int slw() { return (GCOLS / 16) * 8 * bits; }     // words per slice of a group
template <int bits> __host__ __device__ constexpr int stw() { return SPS * slw<bits>(); }           // words per stage
constexpr int BUILD_SCRATCH = SMAX * 4 + 3 * (EMAX + 1) * 2 + 64;
template <int bits> __host__ __device__ constexpr int ring_bytes()
{
    return NSTG * stw<bits>() * 4 > BUILD_SCRATCH ? NSTG * stw<bits>() * 4 : ((BUILD_SCRATCH + 15) / 16) * 16;
}
template <int bits, int RM> __host__ __device__ constexpr int smem_bytes() { return ring_bytes<bits>() + NOB * RM * AST * 2; }

// ---------------------------------------------------------------------------------------------
// mbarrier / named barrier / release-acquire primitives (sm_80+)

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, int count)
{
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(sa(b)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* b)
{
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(sa(b)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity)
{
    asm volatile
    (
        "{\n"
        " .reg .pred p;\n"
        " WAIT:\n"
        " mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n"
        " @!p bra WAIT;\n"
        "}\n" :: "r"(sa(b)), "r"(parity) : "memory"
    );
}

// The barrier completes its phase once every cp.async this thread issued so far has landed
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* b)
{
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(sa(b)) : "memory");
}

__device__ __forceinline__ void cbar_sync() { asm volatile("bar.sync %0, %1;\n" :: "n"(CBAR), "n"(CTHREADS) : "memory"); }

__device__ __forceinline__ int atom_add_acq_rel(int* p, int v)
{
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// ---------------------------------------------------------------------------------------------
// Run table (as coop2's, for this block shape and RM-row runs)

template <int RM>
struct WsTable
{
    int16_t order[SMAX];
    int16_t r_exp[SMAX];
    int16_t r_first[SMAX];
    int8_t r_n[SMAX];
    int16_t slot_e[SMAX];
    int16_t srow[SMAX];         // token rows in order: the shared expert's runs
    int n_runs;
    int scan[NWS];
};

struct WsSync
{
    uint64_t full[NSTG];
    uint64_t empty[NSTG];
    uint64_t ofull[NOB];
    uint64_t oempty[NOB];
    int flag[2];
    unsigned done[2];
    float inv[32];
};

template <int RM>
__device__ __forceinline__ void build_runs_ws(const MoeCoop2Params& p, WsTable<RM>& rt, uint32_t* scratch)
{
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int slots = p.slots;

    if (p.bsz == 1)
    {
        if (warp == 0)
        {
            int n_act = 0;
            for (int b = 0; b < slots; b += 32)
            {
                const int s = b + lane;
                const int e = s < slots ? slot_local(p, s) : -1;
                if (s < slots) rt.slot_e[s] = (int16_t) e;
                const unsigned m = __ballot_sync(0xffffffffu, e >= 0);
                if (e >= 0)
                {
                    const int r = n_act + __popc(m & ((1u << lane) - 1));
                    rt.order[r] = (int16_t) s;
                    rt.r_exp[r] = (int16_t) e;
                    rt.r_first[r] = (int16_t) r;
                    rt.r_n[r] = 1;
                }
                n_act += __popc(m);
            }
            if (lane == 0) rt.n_runs = n_act;
        }
        __syncthreads();
        return;
    }

    int* loc = (int*) scratch;
    int16_t* cnt = (int16_t*) (loc + SMAX);
    int16_t* sbase = cnt + EMAX;
    int16_t* rbase = sbase + EMAX + 1;
    const int NL = p.n_local;

    for (int e = t; e < NL; e += WTHREADS) cnt[e] = 0;
    for (int s = t; s < slots; s += WTHREADS)
    {
        loc[s] = slot_local(p, s);
        rt.slot_e[s] = (int16_t) loc[s];
    }
    __syncthreads();

    if (warp == 0)
    {
        for (int b = 0; b < slots; b += 32)
        {
            const int s = b + lane;
            const int e = s < slots ? loc[s] : -1;
            const unsigned m = __match_any_sync(0xffffffffu, e);
            const int lt = __popc(m & ((1u << lane) - 1));
            const int base = e >= 0 ? cnt[e] : 0;
            __syncwarp();
            if (e >= 0)
            {
                loc[s] = e | ((base + lt) << 16);
                if (lt == 0) cnt[e] = (int16_t) (base + __popc(m));
            }
            __syncwarp();
        }
    }
    __syncthreads();

    const int per = (NL + WTHREADS - 1) / WTHREADS;
    const int e0 = t * per;
    int ls = 0, lr = 0;
    for (int i = 0; i < per; ++i)
        if (e0 + i < NL)
        {
            ls += cnt[e0 + i];
            lr += (cnt[e0 + i] + RM - 1) / RM;
        }
    int v = ls | (lr << 16);
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        const int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) rt.scan[warp] = v;
    __syncthreads();
    int wbase = 0, total = 0;
    for (int w = 0; w < NWS; ++w)
    {
        if (w < warp) wbase += rt.scan[w];
        total += rt.scan[w];
    }
    int ex = wbase + v - (ls | (lr << 16));
    int cs = ex & 0xffff, cr = ex >> 16;
    for (int i = 0; i < per; ++i)
        if (e0 + i < NL)
        {
            const int c = cnt[e0 + i];
            sbase[e0 + i] = (int16_t) cs;
            rbase[e0 + i] = (int16_t) cr;
            cs += c;
            cr += (c + RM - 1) / RM;
        }
    if (t == 0) rt.n_runs = total >> 16;
    __syncthreads();

    for (int s = t; s < slots; s += WTHREADS)
    {
        const int l = loc[s];
        if (l >= 0) rt.order[sbase[l & 0xffff] + (l >> 16)] = (int16_t) s;
    }
    for (int e = t; e < NL; e += WTHREADS)
    {
        const int c = cnt[e];
        for (int j = 0; j * RM < c; ++j)
        {
            const int r = rbase[e] + j;
            rt.r_exp[r] = (int16_t) e;
            rt.r_first[r] = (int16_t) (sbase[e] + j * RM);
            rt.r_n[r] = (int8_t) min(RM, c - j * RM);
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Work items. A step's units are the shared expert's (folded: p.s_ptrs set) and the routed
// experts', each in coop2's decomposition (shared runs: RM consecutive token rows); p.s_first puts
// the shared units first. Every role decodes the same item from the same index, so no unit state
// crosses warps.

constexpr int maxb(int a, int b) { return a > b ? a : b; }

struct StepA { int nproj, ks_a, n_r, ks_as, n_s; };
struct StepB { int ks_b, n_r, ks_bs, n_s; };

struct Item
{
    bool sh;
    int run, proj, group, ks, c0, c1;
    int e;                      // local expert (routed)
    const int16_t* rows;        // slots (routed) or token rows (shared)
    int nrows;
};

template <int RM>
__device__ __forceinline__ void item_rows(const MoeCoop2Params& p, const WsTable<RM>& rt, Item& x)
{
    if (x.sh)
    {
        x.e = -1;
        x.rows = rt.srow + x.run * RM;
        x.nrows = min(RM, p.bsz - x.run * RM);
    }
    else
    {
        x.e = rt.r_exp[x.run];
        x.rows = rt.order + rt.r_first[x.run];
        x.nrows = rt.r_n[x.run];
    }
}

template <int RM>
__device__ __forceinline__ Item item_a(const MoeCoop2Params& p, const WsTable<RM>& rt, const StepA& st, int u)
{
    Item x;
    int v;
    if (p.s_first) { x.sh = u < st.n_s; v = x.sh ? u : u - st.n_s; }
    else { x.sh = u >= st.n_r; v = x.sh ? u - st.n_r : u; }
    const int ksn = x.sh ? st.ks_as : st.ks_a;
    const int cg = x.sh ? p.cg_as : p.cg_a;
    x.ks = v % ksn;
    int r = v / ksn;
    x.group = r % cg;
    r /= cg;
    x.proj = r % st.nproj;
    x.run = r / st.nproj;
    const int chunks = p.Hi / KC;
    x.c0 = chunks * x.ks / ksn;
    x.c1 = chunks * (x.ks + 1) / ksn;
    item_rows<RM>(p, rt, x);
    return x;
}

template <int RM>
__device__ __forceinline__ Item item_b(const MoeCoop2Params& p, const WsTable<RM>& rt, const StepB& st, int u)
{
    Item x;
    int v;
    if (p.s_first) { x.sh = u < st.n_s; v = x.sh ? u : u - st.n_s; }
    else { x.sh = u >= st.n_r; v = x.sh ? u - st.n_r : u; }
    const int ksn = x.sh ? st.ks_bs : st.ks_b;
    x.ks = v % ksn;
    x.group = (v / ksn) % p.cg_b;
    x.run = v / (ksn * p.cg_b);
    x.proj = 0;
    const int chunks = (x.sh ? p.I_s : p.I) / KC;
    x.c0 = chunks * x.ks / ksn;
    x.c1 = chunks * (x.ks + 1) / ksn;
    item_rows<RM>(p, rt, x);
    return x;
}

// Gate/up item's projection: 0 gate, 1 up (ungated: up)
__device__ __forceinline__ bool item_gate(const MoeCoop2Params& p, const Item& x) { return p.gated && x.proj == 0; }

__device__ __forceinline__ const uint32_t* item_trellis_a(const MoeCoop2Params& p, const Item& x)
{
    const bool gate = item_gate(p, x);
    if (x.sh) return (const uint32_t*) p.s_ptrs[gate ? 0 : 3];
    return (const uint32_t*) (gate ? p.g_trellis : p.u_trellis)[x.e];
}

__device__ __forceinline__ const half* item_suh_a(const MoeCoop2Params& p, const Item& x)
{
    const bool gate = item_gate(p, x);
    if (x.sh) return (const half*) p.s_ptrs[gate ? 1 : 4];
    return (const half*) (gate ? p.g_suh : p.u_suh)[x.e];
}

// ---------------------------------------------------------------------------------------------
// Weight producer: every stage of every unit of this block, NSTG ahead of the consumers. STW: the
// ring's stage stride (the wider of the step's two widths)

template <int bits, int STW>
__device__ __forceinline__ void produce_stages
(
    const uint32_t* __restrict__ B32, int ntiles, int group, int c0, int c1,
    uint32_t* ring, WsSync& sy, uint32_t& gs, int lane
)
{
    const int ntg = min(GCOLS / 16, ntiles - group * (GCOLS / 16));
    const int nch = ntg * 2 * bits;                       // 16-byte pieces per slice
    const size_t sstride = (size_t) ntiles * 8 * bits;
    const uint32_t* src = B32 + (size_t) group * (GCOLS / 16) * 8 * bits + (size_t) c0 * SPC * sstride;
    const int nst = (c1 - c0) * STG_PER_CHUNK;
    for (int st = 0; st < nst; ++st, ++gs)
    {
        const int s = gs & (NSTG - 1);
        mbar_wait(&sy.empty[s], ((gs / NSTG) & 1) ^ 1);
        uint32_t* dst = ring + s * STW;
        #pragma unroll
        for (int j = 0; j < SPS; ++j)
        {
            const uint32_t* sj = src + (size_t) (st * SPS + j) * sstride;
            for (int o = lane; o < nch; o += 32)
                cp_async_stream(dst + j * slw<bits>() + o * 4, sj + o * 4);
        }
        cp_async_mbar_arrive(&sy.full[s]);
    }
}

// ---------------------------------------------------------------------------------------------
// Consumer: one unit's chunks against the ring; accumulators for RM rows x 32 columns

template <int bits, int cb, int RM, int STW>
__device__ __forceinline__ void consume_unit
(
    const MoeCoop2Params& p, int ntiles, int group, int c0, int c1, int nrows,
    const uint32_t* ring, const half* obuf, WsSync& sy, uint32_t& gs, uint32_t& go,
    FragC (&acc)[RM / 16][WNT][2]
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const bool wact = group * (GCOLS / 16) + warp * WNT < ntiles && !(p.dbg & 1);
    const bool two = RM == 32 && nrows > 16;
    const int g = lane >> 2;
    const int kq = 2 * (lane & 3);

    #pragma unroll
    for (int h = 0; h < RM / 16; ++h)
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
                #pragma unroll
                for (int j = 0; j < 4; ++j) acc[h][t][f][j] = 0.0f;

    for (int c = c0; c < c1; ++c, ++go)
    {
        const int ob = go & (NOB - 1);
        mbar_wait(&sy.ofull[ob], (go / NOB) & 1);
        const half* ab = obuf + ob * (RM * AST);
        #pragma unroll 1
        for (int stc = 0; stc < STG_PER_CHUNK; ++stc, ++gs)
        {
            const int s = gs & (NSTG - 1);
            mbar_wait(&sy.full[s], (gs / NSTG) & 1);
            if (wact)
            {
                #pragma unroll
                for (int j = 0; j < SPS; ++j)
                {
                    const int kk = (stc * SPS + j) * 16;
                    FragB a01[RM / 16], a23[RM / 16];
                    #pragma unroll
                    for (int h = 0; h < RM / 16; ++h)
                    {
                        if (h == 1 && !two) continue;
                        const half* ar = ab + (h * 16) * AST + kk + kq;
                        a01[h][0] = *((const half2*) (ar + g * AST));
                        a01[h][1] = *((const half2*) (ar + (g + 8) * AST));
                        a23[h][0] = *((const half2*) (ar + g * AST + 8));
                        a23[h][1] = *((const half2*) (ar + (g + 8) * AST + 8));
                    }
                    const uint32_t* sl = ring + s * STW + j * slw<bits>() + warp * WNT * 8 * bits;
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                    {
                        FragB f0, f1;
                        dq_dispatch<bits, cb>(sl + t * 8 * bits, lane << 3, f0, f1);
                        mma_f32(a01[0], a23[0], f0, acc[0][t][0]);
                        mma_f32(a01[0], a23[0], f1, acc[0][t][1]);
                        if (RM == 32 && two)
                        {
                            mma_f32(a01[RM / 16 - 1], a23[RM / 16 - 1], f0, acc[RM / 16 - 1][t][0]);
                            mma_f32(a01[RM / 16 - 1], a23[RM / 16 - 1], f1, acc[RM / 16 - 1][t][1]);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&sy.empty[s]);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&sy.oempty[ob]);
    }
}

// Partial sums: lane holds rows h*16 + g, h*16 + g + 8 and two adjacent columns per (tile, frag)
template <int RM>
__device__ __forceinline__ void store_partials_ws
(
    float* part, int width, int ntiles, int group, const int16_t* rows, int nrows, FragC (&acc)[RM / 16][WNT][2]
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile0 = group * (GCOLS / 16) + warp * WNT;
    if (tile0 >= ntiles) return;
    const int g = lane >> 2;
    #pragma unroll
    for (int h = 0; h < RM / 16; ++h)
    {
        const int r0 = h * 16 + g < nrows ? rows[h * 16 + g] : -1;
        const int r1 = h * 16 + g + 8 < nrows ? rows[h * 16 + g + 8] : -1;
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = (tile0 + t) * 16 + f * 8 + 2 * (lane & 3);
                if (r0 >= 0) *((float2*) (part + (size_t) r0 * width + col)) = make_float2(acc[h][t][f][0], acc[h][t][f][1]);
                if (r1 >= 0) *((float2*) (part + (size_t) r1 * width + col)) = make_float2(acc[h][t][f][2], acc[h][t][f][3]);
            }
    }
}

// ---------------------------------------------------------------------------------------------
// Operand staging

// One row's raw operand piece (4 columns per lane): x as stored (fp16 / bf16), or the fp32 activation
template <bool IS_B> struct RawRow { uint2 v; };
template <> struct RawRow<true> { float4 v; };

template <bool IS_B>
__device__ __forceinline__ RawRow<IS_B> raw_load(const MoeCoop2Params& p, const float* act, int width, int s, int col)
{
    RawRow<IS_B> r;
    if constexpr (IS_B) r.v = __ldcg((const float4*) (act + (size_t) s * width + col));
    else
    {
        // H is a multiple of 4: a lane's four columns are all in range or all out
        const size_t xo = (size_t) (s / p.topk) * p.x_stride + col;
        r.v = col < p.H ? *((const uint2*) (((const half*) p.x) + xo)) : make_uint2(0u, 0u);
    }
    return r;
}

template <bool IS_B>
__device__ __forceinline__ void raw_float(const MoeCoop2Params& p, const RawRow<IS_B>& r, float (&v)[4])
{
    if constexpr (IS_B) { v[0] = r.v.x; v[1] = r.v.y; v[2] = r.v.z; v[3] = r.v.w; }
    else if (p.x_bf16)
    {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.v.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&r.v.y));
        v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
    else
    {
        const half2 a = *reinterpret_cast<const half2*>(&r.v.x);
        const half2 b = *reinterpret_cast<const half2*>(&r.v.y);
        v[0] = __low2float(a); v[1] = __high2float(a); v[2] = __low2float(b); v[3] = __high2float(b);
    }
}

// Stage one chunk's rows r0, r0 + NSW, ... of the unit (this stager's share): RB rows' loads in
// flight while the previous RB are rotated / scaled and stored
template <bool IS_B>
__device__ __forceinline__ void stage_rows
(
    const MoeCoop2Params& p, int c, half* dst, const int16_t* rows, int nrows, const half* suh, const float* act, int width,
    const float* inv, int lane, int r0
)
{
    const int col = c * KC + lane * 4;
    if (p.dbg & 2) return;
    constexpr int RB = 4;
    float sc[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if constexpr (!IS_B)
    {
        const uint2 u = *((const uint2*) (suh + col));
        const half2 a = *reinterpret_cast<const half2*>(&u.x);
        const half2 b = *reinterpret_cast<const half2*>(&u.y);
        sc[0] = __low2float(a); sc[1] = __high2float(a); sc[2] = __low2float(b); sc[3] = __high2float(b);
    }
    RawRow<IS_B> cur[RB], nxt[RB];
    auto load = [&] (int rb, RawRow<IS_B> (&raw)[RB])
    {
        #pragma unroll
        for (int i = 0; i < RB; ++i)
        {
            const int r = rb + i * NSW;
            if (r < nrows) raw[i] = raw_load<IS_B>(p, act, width, rows[r], col);
        }
    };
    load(r0, cur);
    for (int rb = r0; rb < nrows; rb += RB * NSW)
    {
        if (rb + RB * NSW < nrows) load(rb + RB * NSW, nxt);
        #pragma unroll
        for (int i = 0; i < RB; ++i)
        {
            const int r = rb + i * NSW;
            if (r >= nrows) continue;
            float v[4];
            raw_float<IS_B>(p, cur[i], v);
            if constexpr (!IS_B)
            {
                v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                had128(v[0], v[1], v[2], v[3], lane);
            }
            else
            {
                const float s = inv[r];
                v[0] *= s; v[1] *= s; v[2] *= s; v[3] *= s;
            }
            store_h4(dst + r * AST + lane * 4, v[0], v[1], v[2], v[3]);
        }
        #pragma unroll
        for (int i = 0; i < RB; ++i) cur[i] = nxt[i];
    }
}


// Power-of-two activation scale of one row (slot_scale over any peak table): see slot_scale
__device__ __forceinline__ float act_scale(const MoeCoop2Params& p, const float* peak, int nc, int s, int lane)
{
    if (!p.act_auto) return 1.0f;
    float m = 0.0f;
    for (int c = lane; c < nc; c += 32) m = fmaxf(m, __ldcg(peak + (size_t) s * nc + c));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (!(m <= 3.0e38f) || m == 0.0f) return 1.0f;
    return exp2f(fmaxf(ceilf(log2f(m / MOE_COOP2_ACT_AUTO_PEAK)), -96.0f));
}

// ---------------------------------------------------------------------------------------------
// Kernel A: gate/up. The last arrival per (run, group) runs the epilogue for the group's columns

struct EpiA
{
    const float* part;          // the set's partials, projection-major
    size_t pstride;             // one (projection, k segment) block: rows * I
    int I;
    int ks;
    const half* u_svh;
    const half* g_svh;
    const half* d_suh;
    float* act32;
    float* peak;
};

__device__ __forceinline__ void epilogue_a(const MoeCoop2Params& p, const EpiA& ep, const int16_t* rows, int nrows, int group)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, ep.I / KC - group * (GCOLS / KC));
    for (int pr = warp; pr < ((p.dbg & 32) ? 0 : nrows * qn); pr += NCW)
    {
        const int rr = pr / qn;
        const int q = group * (GCOLS / KC) + pr % qn;
        const int s = rows[rr];
        const int col = q * KC + lane * 4;
        const float* pu = ep.part + (size_t) ((p.gated ? 1 : 0) * ep.ks) * ep.pstride + (size_t) s * ep.I + col;
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        for (int k = 0; k < ep.ks; ++k)
        {
            const float4 v = ldcg4(pu + k * ep.pstride);
            u0 += v.x; u1 += v.y; u2 += v.z; u3 += v.w;
        }
        had128(u0, u1, u2, u3, lane);
        scale_h4(ep.u_svh + col, u0, u1, u2, u3);
        float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
        if (p.gated)
        {
            const float* pg = ep.part + (size_t) s * ep.I + col;
            g0 = g1 = g2 = g3 = 0.0f;
            for (int k = 0; k < ep.ks; ++k)
            {
                const float4 v = ldcg4(pg + k * ep.pstride);
                g0 += v.x; g1 += v.y; g2 += v.z; g3 += v.w;
            }
            had128(g0, g1, g2, g3, lane);
            scale_h4(ep.g_svh + col, g0, g1, g2, g3);
        }
        float a0 = act_gate(p.act, p.gated, g0, u0);
        float a1 = act_gate(p.act, p.gated, g1, u1);
        float a2 = act_gate(p.act, p.gated, g2, u2);
        float a3 = act_gate(p.act, p.gated, g3, u3);
        scale_h4(ep.d_suh + col, a0, a1, a2, a3);
        had128(a0, a1, a2, a3, lane);
        *((float4*) (ep.act32 + (size_t) s * ep.I + col)) = make_float4(a0, a1, a2, a3);
        const bool bad = nonfinite4(a0, a1, a2, a3);
        count_ovf(p, bad, lane);
        float m = bad ? INFINITY : fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) ep.peak[(size_t) s * (ep.I / KC) + q] = m;
    }
}

__device__ __forceinline__ void ws_prologue(WsSync& sy)
{
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NSTG; ++i) { mbar_init(&sy.full[i], 32); mbar_init(&sy.empty[i], NCW); }
        for (int i = 0; i < NOB; ++i) { mbar_init(&sy.ofull[i], 32 * NSW); mbar_init(&sy.oempty[i], NCW); }
    }
}

template <int RM>
__device__ __forceinline__ void fill_srow(const MoeCoop2Params& p, WsTable<RM>& rt)
{
    for (int i = threadIdx.x; i < p.bsz; i += WTHREADS) rt.srow[i] = (int16_t) i;
}

template <int bits, int bits_s, int cb, int RM>
__global__ __launch_bounds__(WTHREADS, 2)
void exl3_moe_ws_a_kernel(const MoeCoop2Params p)
{
    constexpr int STW = stw<maxb(bits, bits_s)>();
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* obuf = (half*) (((uint8_t*) smem) + ring_bytes<maxb(bits, bits_s)>());
    __shared__ WsTable<RM> rt;
    __shared__ WsSync sy;

    for (int i = blockIdx.x * WTHREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * WTHREADS) p.ctr_b[i] = 0;
    ws_prologue(sy);
    fill_srow<RM>(p, rt);
    build_runs_ws<RM>(p, rt, ring);                     // ends in a block barrier: the mbarriers are visible

    StepA st;
    st.nproj = p.gated ? 2 : 1;
    st.ks_a = p.ks_a > 0 ? p.ks_a : pick_ks(rt.n_runs * st.nproj * p.cg_a, p.Hi / KC, gridDim.x, p.cap_a / p.slots);
    st.n_r = rt.n_runs * st.nproj * p.cg_a * st.ks_a;
    st.ks_as = 0;
    st.n_s = 0;
    if (p.s_ptrs)
    {
        const int n_sr = (p.bsz + RM - 1) / RM;
        st.ks_as = p.ks_as > 0 ? p.ks_as : pick_ks(n_sr * st.nproj * p.cg_as, p.Hi / KC, gridDim.x, p.cap_s);
        st.n_s = n_sr * st.nproj * p.cg_as * st.ks_as;
    }
    const int n_units = st.n_r + st.n_s;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (warp == WPW)
    {
        uint32_t gs = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Item x = item_a<RM>(p, rt, st, u);
            const uint32_t* B32 = item_trellis_a(p, x);
            const int ntiles = (x.sh ? p.I_s : p.I) / 16;
            if constexpr (bits == bits_s) produce_stages<bits, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
            else if (x.sh) produce_stages<bits_s, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
            else produce_stages<bits, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
        }
        return;
    }
    if (warp >= SPW)
    {
        const int sw = warp - SPW;
        uint32_t go = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Item x = item_a<RM>(p, rt, st, u);
            const half* suh = item_suh_a(p, x);
            for (int c = x.c0; c < x.c1; ++c, ++go)
            {
                const int ob = go & (NOB - 1);
                mbar_wait(&sy.oempty[ob], ((go / NOB) & 1) ^ 1);
                stage_rows<false>(p, c, obuf + ob * (RM * AST), x.rows, x.nrows, suh, nullptr, 0, nullptr, lane, sw);
                mbar_arrive(&sy.ofull[ob]);
            }
        }
        return;
    }

    uint32_t gs = 0, go = 0, ui = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x, ++ui)
    {
        const Item x = item_a<RM>(p, rt, st, u);
        const int I = x.sh ? p.I_s : p.I;
        const int ntiles = I / 16;
        FragC acc[RM / 16][WNT][2];
        if constexpr (bits == bits_s) consume_unit<bits, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);
        else if (x.sh) consume_unit<bits_s, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);
        else consume_unit<bits, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);

        const int ks = x.sh ? st.ks_as : st.ks_a;
        const int nr = x.sh ? p.bsz : p.slots;
        float* pbase = x.sh ? p.part_a_s : p.part_a;
        store_partials_ws<RM>(pbase + (size_t) (x.proj * ks + x.ks) * nr * I, I, ntiles, x.group, x.rows, x.nrows, acc);

        // One arrival per (run, group): consumers meet, one thread releases (and acquires when last)
        cbar_sync();
        if (threadIdx.x == 0)
        {
            int* ctr = x.sh ? p.ctr_a_s + x.run * p.cg_as + x.group : p.ctr_a + x.run * p.cg_a + x.group;
            const bool l = atom_add_acq_rel(ctr, 1) == st.nproj * ks - 1;
            if (l) *ctr = 0;
            sy.flag[ui & 1] = l;
        }
        cbar_sync();
        if (sy.flag[ui & 1])
        {
            EpiA ep;
            ep.part = pbase;
            ep.pstride = (size_t) nr * I;
            ep.I = I;
            ep.ks = ks;
            if (x.sh)
            {
                ep.u_svh = (const half*) p.s_ptrs[5];
                ep.g_svh = (const half*) p.s_ptrs[2];
                ep.d_suh = (const half*) p.s_ptrs[7];
                ep.act32 = p.act32_s;
                ep.peak = p.peak_s;
            }
            else
            {
                ep.u_svh = (const half*) p.u_svh[x.e];
                ep.g_svh = (const half*) p.g_svh[x.e];
                ep.d_suh = (const half*) p.d_suh[x.e];
                ep.act32 = p.act32;
                ep.peak = p.peak;
            }
            epilogue_a(p, ep, x.rows, x.nrows, x.group);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: down; the last arrival per (token row, group) combines the row

// Slots per token held in flight by the combine (top_k 6: one pass; larger top_k loops)
constexpr int WS_CMB = 6;

template <int RM>
__device__ __forceinline__ void combine_rows_ws
(
    const MoeCoop2Params& p, const WsTable<RM>& rt, const StepB& st, const Item& x, unsigned done
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, p.Ho / KC - x.group * (GCOLS / KC));
    const size_t pstride = (size_t) p.slots * p.Ho;
    const int nd = __popc(done);
    for (int it = warp; it < nd * qn; it += NCW)
    {
        const int ri = x.rows[__fns(done, 0, it / qn + 1)];
        const int row = x.sh ? ri : ri / p.topk;
        const int q = x.group * (GCOLS / KC) + it % qn;
        const int col = q * KC + lane * 4;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k0 = 0; k0 < p.topk; k0 += WS_CMB)
        {
            float4 v[WS_CMB];
            uint2 sv[WS_CMB];
            float w[WS_CMB];
            #pragma unroll
            for (int j = 0; j < WS_CMB; ++j)
            {
                const int k = k0 + j;
                const int s = row * p.topk + k;
                const int e = k < p.topk ? rt.slot_e[s] : -1;
                v[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                w[j] = 0.0f;
                if (e >= 0)
                {
                    const float* pb = p.part_b + (size_t) s * p.Ho + col;
                    v[j] = ldcg4(pb);
                    for (int i = 1; i < st.ks_b; ++i)
                    {
                        const float4 t = ldcg4(pb + i * pstride);
                        v[j].x += t.x; v[j].y += t.y; v[j].z += t.z; v[j].w += t.w;
                    }
                    w[j] = __ldcg(p.slot_w + s);
                    sv[j] = *((const uint2*) (((const half*) p.d_svh[e]) + col));
                }
            }
            #pragma unroll
            for (int j = 0; j < WS_CMB; ++j)
            {
                const int k = k0 + j;
                if (k >= p.topk || rt.slot_e[row * p.topk + k] < 0) continue;
                float v0 = v[j].x, v1 = v[j].y, v2 = v[j].z, v3 = v[j].w;
                had128(v0, v1, v2, v3, lane);
                const half2 a = *reinterpret_cast<const half2*>(&sv[j].x);
                const half2 b = *reinterpret_cast<const half2*>(&sv[j].y);
                o0 += w[j] * (v0 * __low2float(a)); o1 += w[j] * (v1 * __high2float(a));
                o2 += w[j] * (v2 * __low2float(b)); o3 += w[j] * (v3 * __high2float(b));
            }
        }
        if (p.s_ptrs)
        {
            // The shared expert's term, last (as the unfolded path adds it): its k segments in
            // order, the output rotation * svh, its activation scale
            const size_t ps = (size_t) p.bsz * p.Ho;
            const float* pb = p.part_b_s + (size_t) row * p.Ho + col;
            float4 t = ldcg4(pb);
            for (int i = 1; i < st.ks_bs; ++i)
            {
                const float4 y = ldcg4(pb + i * ps);
                t.x += y.x; t.y += y.y; t.z += y.z; t.w += y.w;
            }
            had128(t.x, t.y, t.z, t.w, lane);
            scale_h4(((const half*) p.s_ptrs[8]) + col, t.x, t.y, t.z, t.w);
            const float ws_ = __ldcg(p.slot_w_s + row);
            o0 += ws_ * t.x; o1 += ws_ * t.y; o2 += ws_ * t.z; o3 += ws_ * t.w;
        }
        else if (p.sh_out)
        {
            const float* sh = p.sh_out + (size_t) row * p.H + col;
            if (col + 3 < p.H)
            {
                const float4 t = *((const float4*) sh);
                o0 += t.x; o1 += t.y; o2 += t.z; o3 += t.w;
            }
            else
            {
                if (col + 0 < p.H) o0 += sh[0];
                if (col + 1 < p.H) o1 += sh[1];
                if (col + 2 < p.H) o2 += sh[2];
            }
        }
        count_ovf(p, nonfinite4(o0, o1, o2, o3), lane);
        float* dst = p.out + (size_t) row * p.out_stride + col;
        if (col + 3 < p.H_out) *((float4*) dst) = make_float4(o0, o1, o2, o3);
        else
        {
            if (col + 0 < p.H_out) dst[0] = o0;
            if (col + 1 < p.H_out) dst[1] = o1;
            if (col + 2 < p.H_out) dst[2] = o2;
        }
    }
}

template <int bits, int bits_s, int cb, int RM>
__global__ __launch_bounds__(WTHREADS, 2)
void exl3_moe_ws_b_kernel(const MoeCoop2Params p)
{
    constexpr int STW = stw<maxb(bits, bits_s)>();
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* obuf = (half*) (((uint8_t*) smem) + ring_bytes<maxb(bits, bits_s)>());
    __shared__ WsTable<RM> rt;
    __shared__ WsSync sy;

    for (int i = blockIdx.x * WTHREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * WTHREADS) p.ctr_a[i] = 0;
    for (int i = blockIdx.x * WTHREADS + threadIdx.x; i < p.ctr_a_s_len; i += gridDim.x * WTHREADS) p.ctr_a_s[i] = 0;
    // A token with no active slot on this rank: the shared term alone, or zeros (folded: its
    // shared units' arrivals combine the row)
    if (!p.s_ptrs)
        for (int row = blockIdx.x; row < p.bsz; row += gridDim.x)
            if (token_active_slots(p, row) == 0)
                for (int col = threadIdx.x; col < p.H_out; col += WTHREADS)
                    p.out[(size_t) row * p.out_stride + col] = (p.sh_out && col < p.H) ? p.sh_out[(size_t) row * p.H + col] : 0.0f;
    ws_prologue(sy);
    fill_srow<RM>(p, rt);
    build_runs_ws<RM>(p, rt, ring);

    StepB st;
    st.ks_b = p.ks_b > 0 ? p.ks_b : pick_ks(rt.n_runs * p.cg_b, p.I / KC, gridDim.x, p.cap_b / p.slots, p.maxc_b);
    st.n_r = rt.n_runs * p.cg_b * st.ks_b;
    st.ks_bs = 0;
    st.n_s = 0;
    if (p.s_ptrs)
    {
        const int n_sr = (p.bsz + RM - 1) / RM;
        st.ks_bs = p.ks_bs > 0 ? p.ks_bs : pick_ks(n_sr * p.cg_b, p.I_s / KC, gridDim.x, p.cap_s, p.maxc_b);
        st.n_s = n_sr * p.cg_b * st.ks_bs;
    }
    const int n_units = st.n_r + st.n_s;
    const int ntiles = p.Ho / 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (warp == WPW)
    {
        uint32_t gs = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Item x = item_b<RM>(p, rt, st, u);
            const uint32_t* B32 = (const uint32_t*) (x.sh ? p.s_ptrs[6] : p.d_trellis[x.e]);
            if constexpr (bits == bits_s) produce_stages<bits, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
            else if (x.sh) produce_stages<bits_s, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
            else produce_stages<bits, STW>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
        }
        return;
    }
    if (warp >= SPW)
    {
        const int sw = warp - SPW;
        uint32_t go = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Item x = item_b<RM>(p, rt, st, u);
            const float* act = x.sh ? p.act32_s : p.act32;
            const int width = x.sh ? p.I_s : p.I;
            // Per-row inverse activation scale (exact: a power of two); the combine's row weight
            for (int rr = sw; rr < x.nrows; rr += NSW)
            {
                const int s = x.rows[rr];
                const float sc = act_scale(p, x.sh ? p.peak_s : p.peak, width / KC, s, lane);
                if (lane == 0)
                {
                    sy.inv[rr] = 1.0f / sc;
                    if (x.sh) p.slot_w_s[s] = sc;
                    else p.slot_w[s] = slot_rw(p, s) * sc;
                }
            }
            __syncwarp();
            for (int c = x.c0; c < x.c1; ++c, ++go)
            {
                const int ob = go & (NOB - 1);
                mbar_wait(&sy.oempty[ob], ((go / NOB) & 1) ^ 1);
                stage_rows<true>(p, c, obuf + ob * (RM * AST), x.rows, x.nrows, nullptr, act, width, sy.inv, lane, sw);
                mbar_arrive(&sy.ofull[ob]);
            }
            __syncwarp();
        }
        return;
    }

    uint32_t gs = 0, go = 0, ui = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x, ++ui)
    {
        const Item x = item_b<RM>(p, rt, st, u);
        FragC acc[RM / 16][WNT][2];
        if constexpr (bits == bits_s) consume_unit<bits, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);
        else if (x.sh) consume_unit<bits_s, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);
        else consume_unit<bits, cb, RM, STW>(p, ntiles, x.group, x.c0, x.c1, x.nrows, ring, obuf, sy, gs, go, acc);
        float* part = x.sh ? p.part_b_s + (size_t) x.ks * p.bsz * p.Ho : p.part_b + (size_t) x.ks * p.slots * p.Ho;
        store_partials_ws<RM>(part, p.Ho, ntiles, x.group, x.rows, x.nrows, acc);

        // One arrival per (token row, group) for each of the unit's rows: warp 0's lanes, each a
        // release (and an acquire when last); the done mask by ballot
        cbar_sync();
        if (warp == 0)
        {
            bool l = false;
            if (lane < x.nrows)
            {
                const int row = x.sh ? x.rows[lane] : x.rows[lane] / p.topk;
                int na = 0;
                for (int k = 0; k < p.topk; ++k) na += rt.slot_e[row * p.topk + k] >= 0 ? 1 : 0;
                int* ctr = p.ctr_b + row * p.cg_b + x.group;
                l = atom_add_acq_rel(ctr, 1) == na * st.ks_b + st.ks_bs - 1;
                if (l) *ctr = 0;
            }
            const unsigned m = __ballot_sync(0xffffffffu, l);
            if (lane == 0) sy.done[ui & 1] = (p.dbg & 16) ? 0u : m;
        }
        cbar_sync();
        const unsigned done = sy.done[ui & 1];
        if (done) combine_rows_ws<RM>(p, rt, st, x, done);
    }
}

}  // namespace exl3_moe_ws_ns

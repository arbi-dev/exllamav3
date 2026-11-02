#pragma once

// exl3_moe_coop2, warp-specialised mode (ws): the routed mixture as two launches, gate/up (A)
// then down (B), each a single unit kind with coop2's unit decomposition and reduction order (so
// a row's result is bitwise coop2's at the same k split). Per block:
//
//   consumer warps (NCW): decode the trellis and MMA, 32 columns each; accumulators live only here
//   weight producer (1 warp): a block-shared NSTG-stage cp.async ring, signalled through mbarriers
//                    (cp.async.mbarrier.arrive.noinc); it runs ahead across unit edges
//   operand stager (1 warp): the unit's rows as fp16 MMA operands per 128-k chunk (A: x * suh,
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
constexpr int SPW = NCW + 1;                    // operand stager warp
constexpr int NWS = NCW + 2;
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
// Units (coop2's decomposition)

struct Unit
{
    int run, proj, group, ks, c0, c1;
};

__device__ __forceinline__ Unit unit_a(const MoeCoop2Params& p, int u, int ks_a, int nproj)
{
    Unit x;
    x.ks = u % ks_a;
    int r = u / ks_a;
    x.group = r % p.cg_a;
    r /= p.cg_a;
    x.proj = r % nproj;
    x.run = r / nproj;
    const int chunks = p.Hi / KC;
    x.c0 = chunks * x.ks / ks_a;
    x.c1 = chunks * (x.ks + 1) / ks_a;
    return x;
}

__device__ __forceinline__ Unit unit_b(const MoeCoop2Params& p, int u, int ks_b)
{
    Unit x;
    x.ks = u % ks_b;
    x.group = (u / ks_b) % p.cg_b;
    x.run = u / (ks_b * p.cg_b);
    x.proj = 0;
    const int chunks = p.I / KC;
    x.c0 = chunks * x.ks / ks_b;
    x.c1 = chunks * (x.ks + 1) / ks_b;
    return x;
}

// ---------------------------------------------------------------------------------------------
// Weight producer: every stage of every unit of this block, NSTG ahead of the consumers

template <int bits>
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
        uint32_t* dst = ring + s * stw<bits>();
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

template <int bits, int cb, int RM>
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
                    const uint32_t* sl = ring + s * stw<bits>() + j * slw<bits>() + warp * WNT * 8 * bits;
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
// Operand staging of one chunk, one warp: rows [0, nrows) of the unit

__device__ __forceinline__ void x_load4(const MoeCoop2Params& p, int s, int col, float& v0, float& v1, float& v2, float& v3)
{
    const size_t xo = (size_t) (s / p.topk) * p.x_stride;
    v0 = v1 = v2 = v3 = 0.0f;
    if (p.x_bf16)
    {
        const __nv_bfloat16* xr = ((const __nv_bfloat16*) p.x) + xo;
        if (col + 3 < p.H)
        {
            const uint2 u = *((const uint2*) (xr + col));
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            v0 = a.x; v1 = a.y; v2 = b.x; v3 = b.y;
        }
        else
        {
            if (col + 0 < p.H) v0 = __bfloat162float(xr[col + 0]);
            if (col + 1 < p.H) v1 = __bfloat162float(xr[col + 1]);
            if (col + 2 < p.H) v2 = __bfloat162float(xr[col + 2]);
        }
    }
    else
    {
        const half* xr = p.x + xo;
        if (col + 3 < p.H)
        {
            const uint2 u = *((const uint2*) (xr + col));
            const half2 a = *reinterpret_cast<const half2*>(&u.x);
            const half2 b = *reinterpret_cast<const half2*>(&u.y);
            v0 = __low2float(a); v1 = __high2float(a); v2 = __low2float(b); v3 = __high2float(b);
        }
        else
        {
            if (col + 0 < p.H) v0 = __half2float(xr[col + 0]);
            if (col + 1 < p.H) v1 = __half2float(xr[col + 1]);
            if (col + 2 < p.H) v2 = __half2float(xr[col + 2]);
        }
    }
}

template <bool IS_B>
__device__ __forceinline__ void stage_rows
(
    const MoeCoop2Params& p, int c, half* dst, const int16_t* rows, int nrows, const half* suh, const float* inv, int lane
)
{
    const int col = c * KC + lane * 4;
    if (p.dbg & 2) return;
    constexpr int RB = 4;                               // rows in flight
    float sc0 = 1.0f, sc1 = 1.0f, sc2 = 1.0f, sc3 = 1.0f;
    if constexpr (!IS_B)
    {
        const uint2 u = *((const uint2*) (suh + col));
        const half2 a = *reinterpret_cast<const half2*>(&u.x);
        const half2 b = *reinterpret_cast<const half2*>(&u.y);
        sc0 = __low2float(a); sc1 = __high2float(a); sc2 = __low2float(b); sc3 = __high2float(b);
    }
    for (int r0 = 0; r0 < nrows; r0 += RB)
    {
        float v[RB][4];
        #pragma unroll
        for (int i = 0; i < RB; ++i)
        {
            const int r = r0 + i;
            v[i][0] = v[i][1] = v[i][2] = v[i][3] = 0.0f;
            if (r >= nrows) continue;
            const int s = rows[r];
            if constexpr (!IS_B) x_load4(p, s, col, v[i][0], v[i][1], v[i][2], v[i][3]);
            else
            {
                const float4 a = __ldcg((const float4*) (p.act32 + (size_t) s * p.I + col));
                v[i][0] = a.x; v[i][1] = a.y; v[i][2] = a.z; v[i][3] = a.w;
            }
        }
        #pragma unroll
        for (int i = 0; i < RB; ++i)
        {
            const int r = r0 + i;
            if (r >= nrows) continue;
            if constexpr (!IS_B)
            {
                v[i][0] *= sc0; v[i][1] *= sc1; v[i][2] *= sc2; v[i][3] *= sc3;
                had128(v[i][0], v[i][1], v[i][2], v[i][3], lane);
            }
            else
            {
                const float sc = inv[r];
                v[i][0] *= sc; v[i][1] *= sc; v[i][2] *= sc; v[i][3] *= sc;
            }
            store_h4(dst + r * AST + lane * 4, v[i][0], v[i][1], v[i][2], v[i][3]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel A: gate/up

template <int RM>
__device__ __forceinline__ void epilogue_a(const MoeCoop2Params& p, const int16_t* rows, int nrows, int e, int group, int ks_a)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, p.I / KC - group * (GCOLS / KC));
    const size_t pstride = (size_t) p.slots * p.I;
    for (int pr = warp; pr < ((p.dbg & 32) ? 0 : nrows * qn); pr += NCW)
    {
        const int rr = pr / qn;
        const int q = group * (GCOLS / KC) + pr % qn;
        const int s = rows[rr];
        const int col = q * KC + lane * 4;
        const float* pu = p.part_a + (size_t) ((p.gated ? 1 : 0) * ks_a) * pstride + (size_t) s * p.I + col;
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        for (int k = 0; k < ks_a; ++k)
        {
            const float4 v = ldcg4(pu + k * pstride);
            u0 += v.x; u1 += v.y; u2 += v.z; u3 += v.w;
        }
        had128(u0, u1, u2, u3, lane);
        scale_h4(((const half*) p.u_svh[e]) + col, u0, u1, u2, u3);
        float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
        if (p.gated)
        {
            const float* pg = p.part_a + (size_t) s * p.I + col;
            g0 = g1 = g2 = g3 = 0.0f;
            for (int k = 0; k < ks_a; ++k)
            {
                const float4 v = ldcg4(pg + k * pstride);
                g0 += v.x; g1 += v.y; g2 += v.z; g3 += v.w;
            }
            had128(g0, g1, g2, g3, lane);
            scale_h4(((const half*) p.g_svh[e]) + col, g0, g1, g2, g3);
        }
        float a0 = act_gate(p.act, p.gated, g0, u0);
        float a1 = act_gate(p.act, p.gated, g1, u1);
        float a2 = act_gate(p.act, p.gated, g2, u2);
        float a3 = act_gate(p.act, p.gated, g3, u3);
        scale_h4(((const half*) p.d_suh[e]) + col, a0, a1, a2, a3);
        had128(a0, a1, a2, a3, lane);
        *((float4*) (p.act32 + (size_t) s * p.I + col)) = make_float4(a0, a1, a2, a3);
        const bool bad = nonfinite4(a0, a1, a2, a3);
        count_ovf(p, bad, lane);
        float m = bad ? INFINITY : fmaxf(fmaxf(fabsf(a0), fabsf(a1)), fmaxf(fabsf(a2), fabsf(a3)));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) p.peak[(size_t) s * (p.I / KC) + q] = m;
    }
}

template <int bits, int RM>
__device__ __forceinline__ void ws_prologue(WsSync& sy)
{
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NSTG; ++i) { mbar_init(&sy.full[i], 32); mbar_init(&sy.empty[i], NCW); }
        for (int i = 0; i < NOB; ++i) { mbar_init(&sy.ofull[i], 32); mbar_init(&sy.oempty[i], NCW); }
    }
}

template <int bits, int cb, int RM>
__global__ __launch_bounds__(WTHREADS, 2)
void exl3_moe_ws_a_kernel(const MoeCoop2Params p)
{
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* obuf = (half*) (((uint8_t*) smem) + ring_bytes<bits>());
    __shared__ WsTable<RM> rt;
    __shared__ WsSync sy;

    for (int i = blockIdx.x * WTHREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * WTHREADS) p.ctr_b[i] = 0;
    ws_prologue<bits, RM>(sy);
    build_runs_ws<RM>(p, rt, ring);                     // ends in a block barrier: the mbarriers are visible

    const int nproj = p.gated ? 2 : 1;
    const int ks_a = p.ks_a > 0 ? p.ks_a : pick_ks(rt.n_runs * nproj * p.cg_a, p.Hi / KC, gridDim.x, p.cap_a / p.slots);
    const int n_units = rt.n_runs * nproj * p.cg_a * ks_a;
    const int ntiles = p.I / 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (warp == WPW)
    {
        uint32_t gs = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Unit x = unit_a(p, u, ks_a, nproj);
            const bool is_gate = p.gated && x.proj == 0;
            const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[rt.r_exp[x.run]];
            produce_stages<bits>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
        }
        return;
    }
    if (warp == SPW)
    {
        uint32_t go = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Unit x = unit_a(p, u, ks_a, nproj);
            const bool is_gate = p.gated && x.proj == 0;
            const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[rt.r_exp[x.run]];
            const int16_t* rows = rt.order + rt.r_first[x.run];
            const int nrows = rt.r_n[x.run];
            for (int c = x.c0; c < x.c1; ++c, ++go)
            {
                const int ob = go & (NOB - 1);
                mbar_wait(&sy.oempty[ob], ((go / NOB) & 1) ^ 1);
                stage_rows<false>(p, c, obuf + ob * (RM * AST), rows, nrows, suh, nullptr, lane);
                mbar_arrive(&sy.ofull[ob]);
            }
        }
        return;
    }

    uint32_t gs = 0, go = 0, ui = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x, ++ui)
    {
        const Unit x = unit_a(p, u, ks_a, nproj);
        const int16_t* rows = rt.order + rt.r_first[x.run];
        const int nrows = rt.r_n[x.run];
        FragC acc[RM / 16][WNT][2];
        consume_unit<bits, cb, RM>(p, ntiles, x.group, x.c0, x.c1, nrows, ring, obuf, sy, gs, go, acc);
        float* part = p.part_a + (size_t) (x.proj * ks_a + x.ks) * p.slots * p.I;
        store_partials_ws<RM>(part, p.I, ntiles, x.group, rows, nrows, acc);

        // One arrival per (run, group): consumers meet, one thread releases (and acquires when last)
        cbar_sync();
        if (threadIdx.x == 0)
        {
            int* ctr = p.ctr_a + x.run * p.cg_a + x.group;
            const bool l = atom_add_acq_rel(ctr, 1) == nproj * ks_a - 1;
            if (l) *ctr = 0;
            sy.flag[ui & 1] = l;
        }
        cbar_sync();
        if (sy.flag[ui & 1]) epilogue_a<RM>(p, rows, nrows, rt.r_exp[x.run], x.group, ks_a);
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: down; the last arrival per (token row, group) combines the row

template <int RM>
__device__ __forceinline__ void combine_rows_ws
(
    const MoeCoop2Params& p, const WsTable<RM>& rt, int ks_b, const int16_t* rows, unsigned done, int group
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, p.Ho / KC - group * (GCOLS / KC));
    const size_t pstride = (size_t) p.slots * p.Ho;
    const int nd = __popc(done);
    for (int it = warp; it < nd * qn; it += NCW)
    {
        const int row = rows[__fns(done, 0, it / qn + 1)] / p.topk;
        const int q = group * (GCOLS / KC) + it % qn;
        const int col = q * KC + lane * 4;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k0 = 0; k0 < p.topk; k0 += CMB_K)
        {
            float4 v[CMB_K];
            uint2 sv[CMB_K];
            float w[CMB_K];
            #pragma unroll
            for (int j = 0; j < CMB_K; ++j)
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
                    for (int i = 1; i < ks_b; ++i)
                    {
                        const float4 t = ldcg4(pb + i * pstride);
                        v[j].x += t.x; v[j].y += t.y; v[j].z += t.z; v[j].w += t.w;
                    }
                    w[j] = __ldcg(p.slot_w + s);
                    sv[j] = *((const uint2*) (((const half*) p.d_svh[e]) + col));
                }
            }
            #pragma unroll
            for (int j = 0; j < CMB_K; ++j)
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
        if (p.sh_out)
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

template <int bits, int cb, int RM>
__global__ __launch_bounds__(WTHREADS, 2)
void exl3_moe_ws_b_kernel(const MoeCoop2Params p)
{
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* obuf = (half*) (((uint8_t*) smem) + ring_bytes<bits>());
    __shared__ WsTable<RM> rt;
    __shared__ WsSync sy;

    for (int i = blockIdx.x * WTHREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * WTHREADS) p.ctr_a[i] = 0;
    for (int row = blockIdx.x; row < p.bsz; row += gridDim.x)
        if (token_active_slots(p, row) == 0)
            for (int col = threadIdx.x; col < p.H_out; col += WTHREADS)
                p.out[(size_t) row * p.out_stride + col] = (p.sh_out && col < p.H) ? p.sh_out[(size_t) row * p.H + col] : 0.0f;
    ws_prologue<bits, RM>(sy);
    build_runs_ws<RM>(p, rt, ring);

    const int ks_b = p.ks_b > 0 ? p.ks_b : pick_ks(rt.n_runs * p.cg_b, p.I / KC, gridDim.x, p.cap_b / p.slots, p.maxc_b);
    const int n_units = rt.n_runs * p.cg_b * ks_b;
    const int ntiles = p.Ho / 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    if (warp == WPW)
    {
        uint32_t gs = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Unit x = unit_b(p, u, ks_b);
            const uint32_t* B32 = (const uint32_t*) p.d_trellis[rt.r_exp[x.run]];
            produce_stages<bits>(B32, ntiles, x.group, x.c0, x.c1, ring, sy, gs, lane);
        }
        return;
    }
    if (warp == SPW)
    {
        uint32_t go = 0;
        for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        {
            const Unit x = unit_b(p, u, ks_b);
            const int16_t* rows = rt.order + rt.r_first[x.run];
            const int nrows = rt.r_n[x.run];
            // Per-row inverse activation scale (exact: a power of two); the combine's slot weight
            for (int rr = 0; rr < nrows; ++rr)
            {
                const int s = rows[rr];
                const float sc = slot_scale(p, s, lane);
                if (lane == 0)
                {
                    sy.inv[rr] = 1.0f / sc;
                    p.slot_w[s] = slot_rw(p, s) * sc;
                }
            }
            __syncwarp();
            for (int c = x.c0; c < x.c1; ++c, ++go)
            {
                const int ob = go & (NOB - 1);
                mbar_wait(&sy.oempty[ob], ((go / NOB) & 1) ^ 1);
                stage_rows<true>(p, c, obuf + ob * (RM * AST), rows, nrows, nullptr, sy.inv, lane);
                mbar_arrive(&sy.ofull[ob]);
            }
            __syncwarp();
        }
        return;
    }

    uint32_t gs = 0, go = 0, ui = 0;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x, ++ui)
    {
        const Unit x = unit_b(p, u, ks_b);
        const int16_t* rows = rt.order + rt.r_first[x.run];
        const int nrows = rt.r_n[x.run];
        FragC acc[RM / 16][WNT][2];
        consume_unit<bits, cb, RM>(p, ntiles, x.group, x.c0, x.c1, nrows, ring, obuf, sy, gs, go, acc);
        float* part = p.part_b + (size_t) x.ks * p.slots * p.Ho;
        store_partials_ws<RM>(part, p.Ho, ntiles, x.group, rows, nrows, acc);

        // One arrival per (token row, group) for each of the run's slots: warp 0's lanes, each a
        // release (and an acquire when last); the done mask by ballot
        cbar_sync();
        if (warp == 0)
        {
            bool l = false;
            if (lane < nrows)
            {
                const int row = rows[lane] / p.topk;
                int na = 0;
                for (int k = 0; k < p.topk; ++k) na += rt.slot_e[row * p.topk + k] >= 0 ? 1 : 0;
                int* ctr = p.ctr_b + row * p.cg_b + x.group;
                l = atom_add_acq_rel(ctr, 1) == na * ks_b - 1;
                if (l) *ctr = 0;
            }
            const unsigned m = __ballot_sync(0xffffffffu, l);
            if (lane == 0) sy.done[ui & 1] = (p.dbg & 16) ? 0u : m;
        }
        cbar_sync();
        const unsigned done = sy.done[ui & 1];
        if (done) combine_rows_ws<RM>(p, rt, ks_b, rows, done, x.group);
    }
}

}  // namespace exl3_moe_ws_ns

#pragma once

// exl3_moe_coop2 kernels (see exl3_moe_coop2.cuh for the scheme). Instantiated per (K, codebook)
// in comp_units/exl3_moe_coop2_inst_k*.cu.

#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "../compat.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_coop2.cuh"

namespace exl3_moe_coop2_ns {

constexpr int NW = 8;                           // warps per block
constexpr int THREADS = NW * 32;
constexpr int WNT = 2;                          // 16-column tiles per warp
constexpr int GCOLS = NW * WNT * 16;            // columns per group
constexpr int RMAX = 16;                        // slots per run (MMA rows)
constexpr int KC = 128;                         // k chunk: one Hadamard block
constexpr int SPC = KC / 16;                    // k slices per chunk
constexpr int AST = KC + 8;                     // staged operand row stride (halves), bank-padded
constexpr int SMAX = MOE_COOP2_MAX_SLOTS;
constexpr int EMAX = MOE_COOP2_MAX_EXPERTS;
constexpr int RING_BYTES = 4096;                // weight ring per warp
constexpr float R_SCALE = 0.088388347648f;      // 1/sqrt(128)

static_assert(GCOLS == MOE_COOP2_GROUP_COLS, "group width mismatch");

template <int bits> __host__ __device__ constexpr int slice_words() { return WNT * 8 * bits; }
template <int bits> __host__ __device__ constexpr int ring_depth()
{
    return RING_BYTES / (slice_words<bits>() * 4) < 16 ? RING_BYTES / (slice_words<bits>() * 4) : 16;
}
constexpr int RING_WORDS = RING_BYTES / 4;

// Dynamic shared memory: weight rings, two staged operand chunks. The run-table build borrows
// the rings before the first unit
constexpr int SMEM_RING = NW * RING_BYTES;
constexpr int SMEM_ABUF = 2 * RMAX * AST * 2;
constexpr int SMEM_BYTES = SMEM_RING + SMEM_ABUF;
static_assert(SMEM_RING >= SMAX * 4 + 3 * (EMAX + 1) * 2 + 64, "run-table scratch exceeds the rings");
static_assert(SMEM_RING >= NW * 128 * 4, "combine scratch exceeds the rings");

// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_f32(const FragB& a01, const FragB& a23, const FragB& b, FragC& c)
{
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    float* cc = reinterpret_cast<float*>(&c);
    asm volatile
    (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(cc[0]), "+f"(cc[1]), "+f"(cc[2]), "+f"(cc[3])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
}

__device__ __forceinline__ float act_silu(float x) { return x * __fdividef(1.0f, 1.0f + __expf(-x)); }

__device__ __forceinline__ float act_gelu(float x)
{
    const float c = 0.797884560803f;
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_opt(t));
}

// Gated: act(g) * u. Ungated (g = u): relu(u) * u = relu2(u)
__device__ __forceinline__ float act_gate(int act, bool gated, float g, float u)
{
    if (!gated) return fmaxf(0.0f, u) * u;
    if (act == MOE_COOP2_ACT_SILU) return act_silu(g) * u;
    if (act == MOE_COOP2_ACT_GELU) return act_gelu(g) * u;
    const float r = fmaxf(0.0f, g);
    return r * r * u;
}

__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v0 = h0 * R_SCALE; v1 = h1 * R_SCALE; v2 = h2 * R_SCALE; v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void scale_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    const uint2 u = *((const uint2*) p);
    const half2 a = *reinterpret_cast<const half2*>(&u.x);
    const half2 b = *reinterpret_cast<const half2*>(&u.y);
    v0 *= __low2float(a); v1 *= __high2float(a); v2 *= __low2float(b); v3 *= __high2float(b);
}

__device__ __forceinline__ void store_h4(half* p, float v0, float v1, float v2, float v3)
{
    half2 a = __floats2half2_rn(v0, v1);
    half2 b = __floats2half2_rn(v2, v3);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *((uint2*) p) = u;
}

__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg((const float4*) p); }

__device__ __forceinline__ void count_ovf(const MoeCoop2Params& p, bool bad, int lane)
{
    if (!p.ovf) return;
    const unsigned m = __ballot_sync(0xffffffffu, bad);
    if (m && lane == 0) atomicAdd(p.ovf, 1);
}

__device__ __forceinline__ bool nonfinite4(float a, float b, float c, float d)
{
    const float m = 3.0e38f;
    return !(fabsf(a) <= m) || !(fabsf(b) <= m) || !(fabsf(c) <= m) || !(fabsf(d) <= m);
}

// Local expert of a slot, -1 when inactive (zero weight, or another rank's expert)
__device__ __forceinline__ int slot_local(const MoeCoop2Params& p, int s)
{
    int64_t e = p.sel[s];
    if (__half2float(p.rw[s]) == 0.0f) return -1;
    if (p.min_expert >= 0)
    {
        if (e < p.min_expert || e >= p.max_expert) return -1;
        e -= p.min_expert;
    }
    return (int) e;
}

__device__ __forceinline__ int token_active_slots(const MoeCoop2Params& p, int row)
{
    int n = 0;
    for (int k = 0; k < p.topk; ++k) n += slot_local(p, row * p.topk + k) >= 0 ? 1 : 0;
    return n;
}

// Power-of-two scale bringing a slot's activation peak to at most MOE_COOP2_ACT_AUTO_PEAK
// (1 when it already fits). Pure function of the slot's chunk peaks: kernel B's staging and its
// combine compute the same value
__device__ __forceinline__ float slot_scale(const MoeCoop2Params& p, int s, int lane)
{
    if (!p.act_auto) return 1.0f;
    const int nc = p.I / KC;
    float m = 0.0f;
    for (int c = lane; c < nc; c += 32) m = fmaxf(m, __ldcg(p.peak + (size_t) s * nc + c));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (!(m <= 3.0e38f)) return 1.0f;   // non-finite: counted by kernel A, nothing to rescue
    return m > MOE_COOP2_ACT_AUTO_PEAK ? exp2f(ceilf(log2f(m / MOE_COOP2_ACT_AUTO_PEAK))) : 1.0f;
}

// K segments per unit for a step of u1 (run, projection, group) tiles of `chunks` k chunks on a
// grid of G blocks: the split that minimises waves * (chunks per unit + a unit's fixed cost of
// half a chunk), ties to the smaller split. Bounded by the partial-sum capacity. Every block
// computes the same value from the same run table: a pure function of the routing
__host__ __device__ __forceinline__ int pick_ks(int u1, int chunks, int G, int cap)
{
    const int hi = chunks < cap ? chunks : cap;
    int best = 1, best_cost = 0x7fffffff;
    for (int ks = 1; ks <= hi; ++ks)
    {
        const int per = (chunks + ks - 1) / ks;
        const int waves = (u1 * ks + G - 1) / G;
        const int cost = waves * (2 * per + 1);
        if (cost < best_cost) { best_cost = cost; best = ks; }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// Run table: active slots sorted stably by (local expert, slot), cut into runs of at most RMAX
// slots of one expert. Every block builds the same table from the routing. Persistent arrays
// (static shared): order (slot of each sorted position), per run: expert, first sorted position,
// slot count. Scratch in the (still idle) weight rings

struct RunTable
{
    int16_t order[SMAX];
    int16_t r_exp[SMAX];
    int16_t r_first[SMAX];
    int8_t r_n[SMAX];
    int16_t slot_e[SMAX];       // local expert of each slot, -1 inactive
    int n_runs;
    int scan[NW];
};

__device__ __forceinline__ void build_runs(const MoeCoop2Params& p, RunTable& rt, uint32_t* scratch)
{
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int slots = p.slots;

    // bsz 1: top-k picks are distinct, a run per active slot in slot order
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

    int* loc = (int*) scratch;                      // [SMAX] local | rank << 16
    int16_t* cnt = (int16_t*) (loc + SMAX);         // [EMAX] slots per expert
    int16_t* sbase = cnt + EMAX;                    // [EMAX + 1] first sorted position
    int16_t* rbase = sbase + EMAX + 1;              // [EMAX] first run
    const int NL = p.n_local;

    for (int e = t; e < NL; e += THREADS) cnt[e] = 0;
    for (int s = t; s < slots; s += THREADS)
    {
        loc[s] = slot_local(p, s);
        rt.slot_e[s] = (int16_t) loc[s];
    }
    __syncthreads();

    // Stable rank within the expert, one warp over the slots in order
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

    // Exclusive scans over experts of slot counts and run counts (ceil(cnt / RMAX))
    const int per = (NL + THREADS - 1) / THREADS;
    const int e0 = t * per;
    int ls = 0, lr = 0;
    for (int i = 0; i < per; ++i)
        if (e0 + i < NL)
        {
            ls += cnt[e0 + i];
            lr += (cnt[e0 + i] + RMAX - 1) / RMAX;
        }
    // pack (slots, runs) into one int: both <= 512
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
    for (int w = 0; w < NW; ++w)
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
            cr += (c + RMAX - 1) / RMAX;
        }
    if (t == 0) rt.n_runs = total >> 16;
    __syncthreads();

    for (int s = t; s < slots; s += THREADS)
    {
        const int l = loc[s];
        if (l >= 0) rt.order[sbase[l & 0xffff] + (l >> 16)] = (int16_t) s;
    }
    for (int e = t; e < NL; e += THREADS)
    {
        const int c = cnt[e];
        for (int j = 0; j * RMAX < c; ++j)
        {
            const int r = rbase[e] + j;
            rt.r_exp[r] = (int16_t) e;
            rt.r_first[r] = (int16_t) (sbase[e] + j * RMAX);
            rt.r_n[r] = (int8_t) min(RMAX, c - j * RMAX);
        }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Stage one k chunk of the unit's rows as fp16 MMA operand rows: A = rotated input
// (x * suh, Hadamard), B = activation * inverse slot scale

template <bool IS_B>
__device__ __forceinline__ void stage_chunk
(
    const MoeCoop2Params& p, int c, half* dst, const int* rows, int nrows, const half* suh, const float* inv
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int col = c * KC + lane * 4;
    if (p.dbg & 2) return;
    for (int r = warp; r < nrows; r += NW)
    {
        const int s = rows[r];
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        if constexpr (!IS_B)
        {
            const half* xr = p.x + (size_t) (s / p.topk) * p.x_stride;
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
            scale_h4(suh + col, v0, v1, v2, v3);
            had128(v0, v1, v2, v3, lane);
        }
        else
        {
            const float4 a = __ldcg((const float4*) (p.act32 + (size_t) s * p.I + col));
            const float sc = inv[r];
            v0 = a.x * sc; v1 = a.y * sc; v2 = a.z * sc; v3 = a.w * sc;
        }
        store_h4(dst + r * AST + lane * 4, v0, v1, v2, v3);
    }
}

// ---------------------------------------------------------------------------------------------
// One unit's GEMV: rows x (k chunks [c0, c1)) x (the block's 256-column group). Each warp streams
// its 32 columns' trellis words through a private cp.async ring (no block barrier per slice);
// one barrier per 128-k chunk for the shared operand buffer. Accumulates in fp32

struct NoWait { __device__ __forceinline__ void operator()() const {} };

template <int bits, int cb, bool IS_B, typename Wait = NoWait>
__device__ __forceinline__ void unit_gemv
(
    const MoeCoop2Params& p,
    const uint32_t* __restrict__ B32,
    int ntiles,                 // 16-column tiles across the projection's width
    int group,
    int c0, int c1,
    const int* rows, int nrows,
    const half* suh,            // A: the expert's input scale
    const float* inv,           // B: per-row inverse activation scale
    uint32_t* ring, half* abuf,
    int pfd,                    // L2 prefetch distance in slices, 0 = none
    FragC (&acc)[WNT][2],
    const Wait& wait = Wait()   // block-wide: runs after the weight ring is in flight, before the operand is read
)
{
    constexpr int SW = slice_words<bits>();
    constexpr int D = ring_depth<bits>();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile0 = group * (GCOLS / 16) + warp * WNT;
    const bool wact = tile0 < ntiles;
    uint32_t* wring = ring + warp * RING_WORDS;
    const size_t sstride = (size_t) ntiles * 8 * bits;
    const uint32_t* bsrc = B32 + (size_t) tile0 * 8 * bits + (size_t) c0 * SPC * sstride + lane * 4;
    const int ns = (c1 - c0) * SPC;
    const bool cpl = wact && lane < SW / 4;

    // L2 prefetch runs pfd slices ahead of the ring: the smem ring bounds what a warp can hold,
    // not what it can have requested, so the whole unit's weights are in flight from the start
    constexpr int LPS = CEIL_DIVIDE(SW * 4, 128);       // 128-byte lines per slice
    const uint32_t* pbase = B32 + (size_t) tile0 * 8 * bits + (size_t) c0 * SPC * sstride;
    auto prefetch = [&] (int i0, int i1)
    {
        if (!wact) return;
        for (int j = lane; j < (i1 - i0) * LPS; j += 32)
        {
            const int i = i0 + j / LPS;
            const uint32_t* a = pbase + (size_t) i * sstride + (j % LPS) * 32;
            asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
        }
    };
    prefetch(0, min(ns, pfd));

    auto issue = [&] (int i)
    {
        if (cpl && i < ns)
        {
            if (p.dbg & 8) cp_async(wring + (i % D) * SW + lane * 4, bsrc + (size_t) i * sstride);
            else cp_async_stream(wring + (i % D) * SW + lane * 4, bsrc + (size_t) i * sstride);
        }
        cp_async_fence();
    };

    #pragma unroll
    for (int i = 0; i < D - 1; ++i) issue(i);

    wait();

    #pragma unroll
    for (int t = 0; t < WNT; ++t)
        #pragma unroll
        for (int f = 0; f < 2; ++f)
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[t][f][j] = 0.0f;

    stage_chunk<IS_B>(p, c0, abuf, rows, nrows, suh, inv);
    __syncthreads();

    const int g = lane >> 2;
    const int kq = 2 * (lane & 3);
    for (int c = c0; c < c1; ++c)
    {
        const half* ab = abuf + ((c - c0) & 1) * (RMAX * AST);
        if (c + 1 < c1) stage_chunk<IS_B>(p, c + 1, abuf + ((c - c0 + 1) & 1) * (RMAX * AST), rows, nrows, suh, inv);

        #pragma unroll
        for (int j = 0; j < SPC; ++j)
        {
            const int i = (c - c0) * SPC + j;
            cp_async_wait<D - 2>();
            __syncwarp();
            issue(i + D - 1);
            if (pfd > 0 && i + pfd < ns && lane < LPS)
            {
                const uint32_t* a = pbase + (size_t) (i + pfd) * sstride + lane * 32;
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a));
            }
            if (!wact || (p.dbg & 1)) continue;

            FragB a01, a23;
            a01[0] = *((const half2*) (ab + g * AST + j * 16 + kq));
            a01[1] = *((const half2*) (ab + (g + 8) * AST + j * 16 + kq));
            a23[0] = *((const half2*) (ab + g * AST + j * 16 + 8 + kq));
            a23[1] = *((const half2*) (ab + (g + 8) * AST + j * 16 + 8 + kq));

            const uint32_t* sl = wring + (i % D) * SW;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                dq_dispatch<bits, cb>(sl + t * 8 * bits, lane << 3, f0, f1);
                mma_f32(a01, a23, f0, acc[t][0]);
                mma_f32(a01, a23, f1, acc[t][1]);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
}

// Partial sums of a unit: lane holds rows g, g + 8 and two adjacent columns per (tile, frag)
__device__ __forceinline__ void store_partials
(
    float* part, int width, int ntiles, int group, const int* rows, int nrows, FragC (&acc)[WNT][2]
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile0 = group * (GCOLS / 16) + warp * WNT;
    if (tile0 >= ntiles) return;
    const int g = lane >> 2;
    const int r0 = g < nrows ? rows[g] : -1;
    const int r1 = g + 8 < nrows ? rows[g + 8] : -1;
    #pragma unroll
    for (int t = 0; t < WNT; ++t)
        #pragma unroll
        for (int f = 0; f < 2; ++f)
        {
            const int col = (tile0 + t) * 16 + f * 8 + 2 * (lane & 3);
            if (r0 >= 0) *((float2*) (part + (size_t) r0 * width + col)) = make_float2(acc[t][f][0], acc[t][f][1]);
            if (r1 >= 0) *((float2*) (part + (size_t) r1 * width + col)) = make_float2(acc[t][f][2], acc[t][f][3]);
        }
}

// Fence the block's stores, then one arrival; true (block-uniform) for the last arrival
__device__ __forceinline__ bool arrive_last(int* counter, int expected, int* sh_flag)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const bool l = atomicAdd(counter, 1) == expected - 1;
        if (l) *counter = 0;
        *sh_flag = l;
    }
    __syncthreads();
    const bool last = *sh_flag;
    if (last) __threadfence();
    return last;
}

// Unit rows: the run's slots, into shared rows[]
__device__ __forceinline__ void load_run(const RunTable& rt, int run, int* rows, int& nrows)
{
    nrows = rt.r_n[run];
    if (threadIdx.x < nrows) rows[threadIdx.x] = rt.order[rt.r_first[run] + threadIdx.x];
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Gate/up unit u of the step (u < run count * nproj * cg_a * ks_a); the last arrival per (run,
// group) runs the epilogue for the group's columns and, when run_ready is set, signals the run's
// down units (fused kernel)

struct UnitShared
{
    int rows[RMAX];
    float inv[RMAX];
    int flag;
    unsigned done;
    int next;
};

template <int bits, int cb>
__device__ __forceinline__ void a_unit
(
    const MoeCoop2Params& p, const RunTable& rt, int u, int ks_a, int pfd,
    uint32_t* ring, half* abuf, UnitShared& us, int* run_ready
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int chunks = p.Hi / KC;
    const int ntiles = p.I / 16;

    const int ks = u % ks_a;
    int r = u / ks_a;
    const int group = r % p.cg_a;
    r /= p.cg_a;
    const int proj = r % nproj;
    const int run = r / nproj;
    const bool is_gate = p.gated && proj == 0;
    const int e = rt.r_exp[run];
    int nrows;
    load_run(rt, run, us.rows, nrows);

    const int c0 = chunks * ks / ks_a;
    const int c1 = chunks * (ks + 1) / ks_a;
    const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[e];
    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[e];

    FragC acc[WNT][2];
    unit_gemv<bits, cb, false>(p, B32, ntiles, group, c0, c1, us.rows, nrows, suh, nullptr, ring, abuf, pfd, acc);
    float* part = p.part_a + (size_t) (proj * ks_a + ks) * p.slots * p.I;
    store_partials(part, p.I, ntiles, group, us.rows, nrows, acc);

    if (!arrive_last(p.ctr_a + run * p.cg_a + group, nproj * ks_a, &us.flag)) return;

    // Epilogue: (row, 128-chunk) pairs of the group, a warp each
    const int qn = min(GCOLS / KC, p.I / KC - group * (GCOLS / KC));
    const size_t pstride = (size_t) p.slots * p.I;
    for (int pr = warp; pr < ((p.dbg & 32) ? 0 : nrows * qn); pr += NW)
    {
        const int rr = pr / qn;
        const int q = group * (GCOLS / KC) + pr % qn;
        const int s = us.rows[rr];
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
    if (run_ready)
    {
        // Publish the group's activation columns to the run's down units
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(run_ready + run, 1);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Kernel A: gate/up units, epilogue by the last arrival per (run, group)

template <int bits, int cb>
__global__ __launch_bounds__(THREADS, 2)
void exl3_moe_coop2_a_kernel(const MoeCoop2Params p)
{
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* abuf = (half*) (smem + NW * RING_WORDS);
    __shared__ RunTable rt;
    __shared__ UnitShared us;

    // Reset the down stage's counters (its previous use finished in stream order)
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS) p.ctr_b[i] = 0;

    build_runs(p, rt, ring);

    const int nproj = p.gated ? 2 : 1;
    const int ks_a = p.ks_a > 0 ? p.ks_a : pick_ks(rt.n_runs * nproj * p.cg_a, p.Hi / KC, gridDim.x, p.cap_a / p.slots);
    const int n_units = rt.n_runs * nproj * p.cg_a * ks_a;
    // A single-wave step streams each unit's weights through L2 prefetches ahead of the ring
    // (several units per block: the prefetches only add traffic)
    const int pfd = (n_units <= (int) gridDim.x && !(p.dbg & 4)) ? p.pfd : 0;

    for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        a_unit<bits, cb>(p, rt, u, ks_a, pfd, ring, abuf, us, nullptr);
}


// ---------------------------------------------------------------------------------------------
// Kernel B: down units; the last arrival per (token row, group) combines the row

__device__ __forceinline__ void combine_row(const MoeCoop2Params& p, int ks_b, int row, int group, float* sh_part)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, p.Ho / KC - group * (GCOLS / KC));
    const int qi = warp & 1;
    const int q = group * (GCOLS / KC) + qi;
    const int col = q * KC + lane * 4;
    const size_t pstride = (size_t) p.slots * p.Ho;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    if (qi < qn)
    {
        for (int k = warp >> 1; k < p.topk; k += NW / 2)
        {
            const int s = row * p.topk + k;
            const int e = slot_local(p, s);
            if (e < 0) continue;
            const float* pb = p.part_b + (size_t) s * p.Ho + col;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int j = 0; j < ks_b; ++j)
            {
                const float4 v = ldcg4(pb + j * pstride);
                v0 += v.x; v1 += v.y; v2 += v.z; v3 += v.w;
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) p.d_svh[e]) + col, v0, v1, v2, v3);
            const float w = __half2float(p.rw[s]) * slot_scale(p, s, lane);
            o0 += w * v0; o1 += w * v1; o2 += w * v2; o3 += w * v3;
        }
    }
    float* part = sh_part + warp * KC + lane * 4;
    part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
    __syncthreads();
    if (warp < 2 && qi < qn)
    {
        o0 = o1 = o2 = o3 = 0.0f;
        #pragma unroll
        for (int m = 0; m < NW / 2; ++m)
        {
            const float* src = sh_part + ((m << 1) | qi) * KC + lane * 4;
            o0 += src[0]; o1 += src[1]; o2 += src[2]; o3 += src[3];
        }
        if (p.sh_out)
        {
            const float* sh = p.sh_out + (size_t) row * p.H + col;
            if (col + 0 < p.H) o0 += sh[0];
            if (col + 1 < p.H) o1 += sh[1];
            if (col + 2 < p.H) o2 += sh[2];
            if (col + 3 < p.H) o3 += sh[3];
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
    __syncthreads();
}

// Fast combine (slot_w scratch present): a warp per (done row, 128-column chunk) sums the row's
// slots in slot order, every slot's loads issued before any is used. Slot activity from the run
// table, the slot's routing weight times its activation scale from slot_w (written by the slot's
// down units before their arrivals)
constexpr int CMB_K = 8;        // slots per token held in flight (larger top_k loops)

__device__ __forceinline__ void combine_rows
(
    const MoeCoop2Params& p, const RunTable& rt, int ks_b, const int* rows, int nrows, unsigned done, int group
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int qn = min(GCOLS / KC, p.Ho / KC - group * (GCOLS / KC));
    const size_t pstride = (size_t) p.slots * p.Ho;
    const int nd = __popc(done);
    for (int it = warp; it < nd * qn; it += NW)
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

// A token with no active slot on this rank: its row is the shared term alone, or zeros
__device__ __forceinline__ void write_empty_row(const MoeCoop2Params& p, int row)
{
    for (int col = threadIdx.x; col < p.H_out; col += THREADS)
        p.out[(size_t) row * p.out_stride + col] = (p.sh_out && col < p.H) ? p.sh_out[(size_t) row * p.H + col] : 0.0f;
}
// Down unit u of the step (u < run count * cg_b * ks_b). wait_run (fused kernel): the run's
// activation is published through run_ready, awaited once the unit's weights are in flight
template <int bits, int cb>
__device__ __forceinline__ void b_unit
(
    const MoeCoop2Params& p, const RunTable& rt, int u, int ks_b, int pfd,
    uint32_t* ring, half* abuf, UnitShared& us, const int* run_ready
)
{
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int chunks = p.I / KC;
    const int ntiles = p.Ho / 16;

    const int ks = u % ks_b;
    const int group = (u / ks_b) % p.cg_b;
    const int run = u / (ks_b * p.cg_b);
    const int e = rt.r_exp[run];
    int nrows;
    load_run(rt, run, us.rows, nrows);

    // Per-row inverse activation scale; after the run's activation is complete
    auto scales = [&] ()
    {
        if (run_ready)
        {
            if (threadIdx.x == 0)
            {
                const int want = p.cg_a;
                while (true)
                {
                    int v;
                    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(run_ready + run) : "memory");
                    if (v >= want) break;
                    __nanosleep(64);
                }
            }
            __syncthreads();
        }
        for (int rr = warp; rr < nrows; rr += NW)
        {
            const int s = us.rows[rr];
            const float sc = slot_scale(p, s, lane);
            if (lane == 0)
            {
                us.inv[rr] = 1.0f / sc;                 // exact: a power of two
                if (p.slot_w) p.slot_w[s] = __half2float(p.rw[s]) * sc;
            }
        }
        __syncthreads();
    };

    const int c0 = chunks * ks / ks_b;
    const int c1 = chunks * (ks + 1) / ks_b;
    const uint32_t* B32 = (const uint32_t*) p.d_trellis[e];

    FragC acc[WNT][2];
    if (run_ready)
        unit_gemv<bits, cb, true>(p, B32, ntiles, group, c0, c1, us.rows, nrows, nullptr, us.inv, ring, abuf, pfd, acc, scales);
    else
    {
        scales();
        unit_gemv<bits, cb, true>(p, B32, ntiles, group, c0, c1, us.rows, nrows, nullptr, us.inv, ring, abuf, pfd, acc);
    }
    float* part = p.part_b + (size_t) ks * p.slots * p.Ho;
    store_partials(part, p.Ho, ntiles, group, us.rows, nrows, acc);

    // One arrival per (token row, group) for each of the run's slots (distinct rows)
    __threadfence();
    if (threadIdx.x == 0) us.done = 0;
    __syncthreads();
    if (threadIdx.x < nrows)
    {
        const int row = us.rows[threadIdx.x] / p.topk;
        int na = 0;
        for (int k = 0; k < p.topk; ++k) na += rt.slot_e[row * p.topk + k] >= 0 ? 1 : 0;
        const int expected = na * ks_b;
        if (atomicAdd(p.ctr_b + row * p.cg_b + group, 1) == expected - 1)
        {
            p.ctr_b[row * p.cg_b + group] = 0;
            atomicOr(&us.done, 1u << threadIdx.x);
        }
    }
    __syncthreads();
    const unsigned done = (p.dbg & 16) ? 0u : us.done;
    if (done)
    {
        __threadfence();
        if (p.slot_w) combine_rows(p, rt, ks_b, us.rows, nrows, done, group);
        else
            for (int rr = 0; rr < nrows; ++rr)
                if (done & (1u << rr))
                    combine_row(p, ks_b, us.rows[rr] / p.topk, group, (float*) ring);
    }
    __syncthreads();
}

template <int bits, int cb>
__global__ __launch_bounds__(THREADS, 2)
void exl3_moe_coop2_b_kernel(const MoeCoop2Params p)
{
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* abuf = (half*) (smem + NW * RING_WORDS);
    __shared__ RunTable rt;
    __shared__ UnitShared us;

    // Reset the gate/up stage's counters for the next call
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS) p.ctr_a[i] = 0;

    for (int row = blockIdx.x; row < p.bsz; row += gridDim.x)
        if (token_active_slots(p, row) == 0) write_empty_row(p, row);

    build_runs(p, rt, ring);

    const int ks_b = p.ks_b > 0 ? p.ks_b : pick_ks(rt.n_runs * p.cg_b, p.I / KC, gridDim.x, p.cap_b / p.slots);
    const int n_units = rt.n_runs * p.cg_b * ks_b;
    for (int u = blockIdx.x; u < n_units; u += gridDim.x)
        b_unit<bits, cb>(p, rt, u, ks_b, 0, ring, abuf, us, nullptr);
}

// ---------------------------------------------------------------------------------------------
// Fused kernel (one launch, gate/up and down at one bit width): blocks draw units from a ticket
// queue, every gate/up unit before any down unit. A down unit puts its weights in flight (ring
// and L2), then waits for its run's activation (run_ready: one count per finished (run, group)
// epilogue). A block only ever waits holding down tickets, and every gate/up ticket is held by a
// running block that never waits, so the queue cannot deadlock whatever the residency. The last
// block out zeroes the queue and run_ready for the next call; the arrival counters reset
// themselves. Deterministic: the unit decomposition and every reduction order are those of the
// two-launch kernels, independent of which block runs a unit.

template <int bits, int cb>
__global__ __launch_bounds__(THREADS, 2)
void exl3_moe_coop2_f_kernel(const MoeCoop2Params p)
{
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* ring = smem;
    half* abuf = (half*) (smem + NW * RING_WORDS);
    __shared__ RunTable rt;
    __shared__ UnitShared us;
    int* ticket = p.fctr;
    int* exits = p.fctr + 1;
    int* run_ready = p.fctr + 2;

    if (threadIdx.x == 0) us.next = atomicAdd(ticket, 1);

    for (int row = blockIdx.x; row < p.bsz; row += gridDim.x)
        if (token_active_slots(p, row) == 0) write_empty_row(p, row);

    build_runs(p, rt, ring);

    const int nproj = p.gated ? 2 : 1;
    const int ks_a = p.ks_a > 0 ? p.ks_a : pick_ks(rt.n_runs * nproj * p.cg_a, p.Hi / KC, gridDim.x, p.cap_a / p.slots);
    const int ks_b = p.ks_b > 0 ? p.ks_b : pick_ks(rt.n_runs * p.cg_b, p.I / KC, gridDim.x, p.cap_b / p.slots);
    const int n_a = rt.n_runs * nproj * p.cg_a * ks_a;
    const int n_b = rt.n_runs * p.cg_b * ks_b;
    const int pfd_a = (n_a <= (int) gridDim.x && !(p.dbg & 4)) ? p.pfd : 0;
    const int pfd_b = (p.dbg & 4) ? 0 : p.pfd_b;

    int u = us.next;            // published by build_runs' barriers
    while (u < n_a + n_b)
    {
        __syncthreads();
        if (threadIdx.x == 0) us.next = atomicAdd(ticket, 1);
        // us.next is read after the unit's own barriers
        if (u < n_a) a_unit<bits, cb>(p, rt, u, ks_a, pfd_a, ring, abuf, us, run_ready);
        else b_unit<bits, cb>(p, rt, u - n_a, ks_b, pfd_b, ring, abuf, us, run_ready);
        u = us.next;
    }

    // Last block out: every unit of the call is done; zero the queue and the run flags
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        us.flag = atomicAdd(exits, 1) == (int) gridDim.x - 1;
    }
    __syncthreads();
    if (us.flag)
    {
        for (int i = threadIdx.x; i < rt.n_runs; i += THREADS) run_ready[i] = 0;
        if (threadIdx.x == 0) { *ticket = 0; *exits = 0; }
    }
}

}  // namespace exl3_moe_coop2_ns

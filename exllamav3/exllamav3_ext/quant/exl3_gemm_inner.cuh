#pragma once

#include "../ptx.cuh"
#include "exl3_devctx.cuh"

// Constants
#define EXL3_GEMM_BASE_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

#include "exl3_dq.cuh"

// On GA10x, HMMA with fp32 accumulation runs at half rate and dominates the m=1 (decode-bound) case.
// Accumulate MMA results in fp16 instead and fold into the fp32 accumulators once per k-slice: ~14%
// faster at bsz 1 on RTX 3090 together with the codebook.cuh IMUL change (see benchmarks/exl3_m1_bench).
// Max observed error vs fp32 accumulation is ~1% of output RMS at k=4096, well below quantization noise.
// Only enabled for sm_86 for now; unvalidated on other archs where fp32-acc HMMA is also half rate.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860)
    #define EXL3_GEMM_H_ACC 1
#else
    #define EXL3_GEMM_H_ACC 0
#endif

template<EXL3_GEMM_T_ARGS, bool shmem_out_had, bool fixup_capable>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    const int size_n_b,
    int* __restrict__ locks,
    const half* post_scale
)
{
    const int TILEBLOCKS_M = TILESIZE_M / 16;
    const int TILEBLOCKS_K = TILESIZE_K / 16;
    const int TILEBLOCKS_N = TILESIZE_N / 16;
    // const int FRAGS_M = TILEBLOCKS_M;
    const int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    const int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                         // in halfs
    const int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;   // in uint16s
    // Every store/add pair in the k-reduction gets its own staging region. store() ends with a
    // barrier but add() does not, so a region shared between two pairs would let the later
    // pair's store overwrite slots the earlier pair's add is still reading. The sequences below
    // run TILEBLOCKS_K - 1 pairs per row block. The output-hadamard staging holds one 16-row
    // block at a time and so does not scale with TILESIZE_M
    const int sh_c_red_stride = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP;
    const int sh_c_red_pairs = TILEBLOCKS_K > 1 ? TILEBLOCKS_K - 1 : 0;
    const int sh_c_size = MAX  // in floats
    (
        TILEBLOCKS_M * sh_c_red_pairs * sh_c_red_stride,
        shmem_out_had ? TILESIZE_N * 16 : 0
    );

    // XOR-swizzle constants for bank-conflict-free A fragment loads
    // col_swizzled = col ^ ((row >> SHIFT) & MASK)
    const int A_COLS = TILESIZE_K / 8;                                            // int4 columns per row
    const int A_SWIZZLE_MASK = A_COLS - 1;
    const int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    // Sanity checks
    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M % 16 == 0, "Invalid kernel params");                  // size_m <= TILESIZE_M
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    // The A-fragment XOR swizzle indexes row m as m * A_COLS + (k ^ x). It only stays inside
    // the row when A_COLS is a power of two; otherwise the swizzled column runs past the row
    // and both the store and the ldsm4 read alias neighbouring rows
    static_assert((A_COLS & (A_COLS - 1)) == 0, "Invalid kernel params (TILESIZE_K / 8 must be a power of two)");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
        "Invalid kernel params (insufficient shared memory for shape)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    // Thread index
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    // Dimensions
    //int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    int tiles_k = size_k / TILESIZE_K;
    int tiles_n = size_n / TILESIZE_N;
    //int blocks_m = 1;
    //int blocks_k = tiles_k * TILEBLOCKS_K;
    int blocks_n = tiles_n * TILEBLOCKS_N;
    // B's row pitch along k is the width of the STORED weight, which is size_n only when the
    // caller computes every output column. A caller that asks for a leading window of a wider
    // weight passes the stored width here and the tile walk above still spans size_n
    int blocks_n_b = size_n_b / 16;

    // Start and end index of current slice, must span at least one tile
    int num_slices = gridDim.x;
    int slice_beg = tiles_k * tiles_n * blockIdx.x / num_slices;
    int slice_end = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    // Parallel fixup eligibility. Uniform across the grid -- every term is a launch constant or a
    // compile-time tile constant, so no two CTAs of one launch can disagree about which reduction
    // they are running, which is what makes a deadlock on the arrival counter impossible.
    // `run_bound <= tiles_k` is what makes two staging slots per CTA enough: a run no longer than
    // one column straddles at most one column boundary, so a CTA stages at most one partial per
    // column and the two columns it can touch have different parity.
    //
    // Which INSTANTIATIONS carry the staging path. Compile-time, so an ineligible shape emits
    // none of it and pays none of its register cost: two slots per CTA at a compile-time grid
    // bound must fit the arena budget. That is the only gate -- a VRAM bound, not a shape list.
    //
    // What it costs where it IS compiled, measured with ptxas rather than assumed
    // (docs/receipts/exl3-streamk-fixup-ptxas.txt): the reducer holds every accumulator live
    // across a variable-trip-count loop of global loads, which is irreducible for any fixup that
    // folds L partials into registers. On the target's own tile that is 122 -> 128 registers with
    // no spill. Tiles already pinned at the 128-register wall by a 512-thread block cannot be
    // given more registers and take it as spill instead, +32 to +44 bytes, in an epilogue that
    // runs once per column rather than in the mainloop.
    constexpr bool fixup_shape_ok = fixup_capable &&
        2ull * EXL3_GEMM_FIXUP_MAX_SLICES * TILESIZE_M * TILESIZE_N * 4
            <= (unsigned long long) EXL3_GEMM_FIXUP_BYTES;
    // Computed under the same guard as everything else it feeds: left outside, its address
    // arithmetic alone cost an ineligible instantiation 2 registers, and "the gate leaves the
    // rest of the tree untouched" then had a counter-example in its own receipt
    float* fixup_arena = fixup_shape_ok ? (float*) (locks + EXL3_GEMM_FIXUP_OFFSET) : nullptr;
    bool use_fixup = false;
    if constexpr (fixup_shape_ok)
    {
        int run_bound = CEIL_DIVIDE(tiles_k * tiles_n, num_slices);
        use_fixup =
            locks[EXL3_GEMM_FIXUP_ENABLE_OFFSET] != 0 &&
            run_bound <= tiles_k &&
            num_slices <= EXL3_GEMM_FIXUP_MAX_SLICES;
    }

    auto index_m = [&] (int slice_i) { return 0; }; //blockIdx.y; };
    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    // Batch dimension
    // int slice_m = index_m(slice_beg);
    // int max_m = MIN(size_m - slice_m * TILESIZE_M, TILESIZE_M);
    const int slice_m = 0;

    // Pipe 0, global A, B tile and shared A, B tile
    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    const int load_a_iters = CEIL_DIVIDE(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i)
    {
        int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m;
    }

    int gl_b_stride_k = blocks_n_b * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    const int load_b_iters = CEIL_DIVIDE(sh0_b_stride_k / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i)
    {
        int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n_b * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
            gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // Pipe 1, shared A, B tile and registers
    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;

    half* sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k)
        {
            slice1_k = 0;
            slice1_n++;
        }
    };

    // Pipe 2
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;

    int gl_c_stride_n = TILESIZE_N;
    int gl_c_stride_m = TILESIZE_M * size_n;

    half* gl_c_ptr_16 = ((half*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    register FragA frag_a[FRAG_STAGES][TILEBLOCKS_M];
    register FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    register FragC frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #if EXL3_GEMM_H_ACC
        register FragC_h frag_c_h[TILEBLOCKS_M][FRAGS_N_PER_WARP];
    #endif

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32)
                gl_c_ptr_32 += gl_c_stride_n;
            else
                gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    // Schedule load of the next A, B tiles to shared memory and advance the pipeline
    auto async_load_gl = [&] ()
    {
        if (sub_k)
        {
            cp_async_fence();
            return;
        }

        if (slice0_iters)
        {
            // Copy tile from row-major A matrix (XOR-swizzled for bank-conflict-free ldmatrix)
            {
                const int4* gl = (const int4*) gl_a_ptr;
                int4* sh = (int4*) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i)
                {
                    if (pred_a_gl[i]) cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                }
            }

            // Copy tile of 256-element blocks from quantized B matrix
            {
                const int4* gl = (const int4*) gl_b_ptr;
                int4* sh = (int4*) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i)
                {
                    // cp_async_pred(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i], pred_b_gl[i]);
                    if (pred_b_gl[i]) cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                }
            }
            advance0();
        }

        // Sync and advance
        cp_async_fence();
    };

    // Load fragments
    // Ref. for fragment layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A fragments (XOR-swizzled shared memory layout), one per row block. The B fragments
        // below are decoded once and reused across all of them, which is the point of TILESIZE_M
        {
            int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                int R = r + m * 16;
                int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf][m], (int4*) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }

        // B fragments
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2)
        {
            int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t* shb = (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);

            dq_dispatch<bits, cb>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }

        __syncthreads();
        advance1();
    };

    // Clear C fragments
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n] = {};
        #if EXL3_GEMM_H_ACC
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                    frag_c_h[m][n] = {};
        #endif
    };

    // Threadblock reduction
    auto threadblock_reduce = [&] ()
    {
        auto store = [&] (int i, int m, int slot)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + slot * sh_c_red_stride + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) *sh_red++ = frag_c[m][n][j];
                }
            }
            __syncthreads();
        };

        auto add = [&] (int i, int m, int slot)
        {
            if (sub_k == i)
            {
                float* sh_red = sh_c + slot * sh_c_red_stride + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[m][n][j] += *sh_red++;
                }
            }
        };

        auto store_small = [&] (int i, int m, int slot)
        {
            if (sub_k == i && m * 16 + lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + slot * sh_c_red_stride + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    *sh_red++ = frag_c[m][n][0];
                    *sh_red++ = frag_c[m][n][1];
                }
            }
            __syncthreads();
        };

        auto add_small = [&] (int i, int m, int slot)
        {
            if (sub_k == i && m * 16 + lane_id / 4 < size_m)
            {
                float* sh_red = sh_c + slot * sh_c_red_stride + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                {
                    frag_c[m][n][0] += *sh_red++;
                    frag_c[m][n][1] += *sh_red++;
                }
            }
        };

        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            const int s0 = m * sh_c_red_pairs;
            if (size_m - m * 16 <= 8)
            {
                if constexpr (TILEBLOCKS_K == 2)
                {
                    store_small(1, m, s0);
                    add_small(0, m, s0);
                }
                if constexpr (TILEBLOCKS_K == 3)
                {
                    store_small(1, m, s0);
                    add_small(0, m, s0);
                    store_small(2, m, s0 + 1);
                    add_small(0, m, s0 + 1);
                }
                if constexpr (TILEBLOCKS_K == 4)
                {
                    store_small(3, m, s0);
                    add_small(2, m, s0);
                    store_small(1, m, s0 + 1);
                    add_small(0, m, s0 + 1);
                    store_small(2, m, s0 + 2);
                    add_small(0, m, s0 + 2);
                }
            }
            else
            {
                if constexpr (TILEBLOCKS_K == 2)
                {
                    store(1, m, s0);
                    add(0, m, s0);
                }
                if constexpr (TILEBLOCKS_K == 3)
                {
                    store(1, m, s0);
                    add(0, m, s0);
                    store(2, m, s0 + 1);
                    add(0, m, s0 + 1);
                }
                if constexpr (TILEBLOCKS_K == 4)
                {
                    store(3, m, s0);
                    add(2, m, s0);
                    store(1, m, s0 + 1);
                    add(0, m, s0 + 1);
                    store(2, m, s0 + 2);
                    add(0, m, s0 + 2);
                }
            }
        }
    };

    // Pre-hadamard: Write final output tile to shmem
    auto write_sum_tile_sh = [&](int m)
    {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        const int r0 = lane_id / 4;
        const int r1 = r0 + 8;
        if (m * 16 + r0 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r0 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[m][n][0];
                *c_ptr++ = frag_c[m][n][1];
            }
        }
        if (m * 16 + r1 < size_m)
        {
            const int c = (lane_id % 4) * 2;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float* c_ptr = ((float*) sh_c) + r1 * TILESIZE_N + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[m][n][2];
                *c_ptr++ = frag_c[m][n][3];
            }
        }
    };

    // Copy output tile to global with hadamard transform and out scale
    auto output_had_sh_gl = [&](int m)
    {
        int rows = TILEBLOCKS_M == 1 ? size_m : MIN(size_m - m * 16, 16);
        int sh_warp = warp_id;
        constexpr int active_warps = EXL3_GEMM_BASE_THREADS / 32;
        for (;; sh_warp += active_warps)
        {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= rows) break;

            const float* had_in = sh_c + row * TILESIZE_N + col * 128;
            const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;

            if constexpr (c_fp32)
            {
                float* had_out = gl_c_ptr_32 + (m * 16 + row) * size_n + col * 128;
                had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
            else
            {
                half* had_out = gl_c_ptr_16 + (m * 16 + row) * size_n + col * 128;
                had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto read_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = m * 16 + lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    frag_c[m][n][0] += *c_ptr++;
                    frag_c[m][n][1] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[m][n][0] += interm.x;
                    frag_c[m][n][1] += interm.y;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    frag_c[m][n][2] += *c_ptr++;
                    frag_c[m][n][3] += *c_ptr++;
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    float2 interm = __half22float2(*c_ptr);
                    frag_c[m][n][2] += interm.x;
                    frag_c[m][n][3] += interm.y;
                }
            }
        }
    };

    auto write_sum_gl = [&]()
    {
        int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            int r0 = m * 16 + lane_id / 4;
            int r1 = r0 + 8;
            int c = (lane_id % 4) * 2;
            if (r0 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r0 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][0];
                    *c_ptr++ = frag_c[m][n][1];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r0 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[m][n][0], frag_c[m][n][1]);
                    *c_ptr = sum;
                }
            }
            if (r1 < size_m)
            {
                if constexpr (c_fp32)
                {
                    float* c_ptr = gl_c_ptr_32 + r1 * size_n + (n0 + n) * 8 + c;
                    *c_ptr++ = frag_c[m][n][2];
                    *c_ptr++ = frag_c[m][n][3];
                }
                else
                {
                    half2* c_ptr = (half2*) (gl_c_ptr_16 + r1 * size_n + (n0 + n) * 8 + c);
                    half2 sum = __floats2half2_rn(frag_c[m][n][2], frag_c[m][n][3]);
                    *c_ptr = sum;
                }
            }
        }
    };

    // Stage this CTA's partial into its own arena slot, and fold a slot back.
    //
    // FLAT, thread-contiguous -- the same layout `threadblock_reduce` already stages through
    // shared memory, not C's. A staging slot is private to the fixup: the CTA that reads it is
    // reading what a CTA wrote with the identical indexing, so it carries no layout obligation
    // whatever. Addressing it like C (`r * size_n + n * 8 + c`, a multiply and a scattered offset
    // per fragment element) is what makes a second output path expensive in this register-bound
    // epilogue -- measured at +40 to +60 registers, spilling four served shapes. This is one
    // pointer and a `+= 4`.
    //
    // Stores are still row-masked, so the traffic is `size_m` rows and not `TILESIZE_M`: at M=1
    // in a 16-row tile a dense dump would move 16x the bytes the chain moves, which on this
    // geometry is 60% of the trellis read and would eat the win outright. Slots a thread skips
    // are never read back, because the reader evaluates the same predicate.
    auto stage_base = [&] (int b, int m) -> float*
    {
        return fixup_arena
             + (size_t) (2 * b + (slice2_n & 1)) * (TILESIZE_M * TILESIZE_N)
             + m * (4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP)
             + (FRAGS_N_PER_WARP * 4) * t;
    };

    auto write_stage = [&] (int b)
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int r0 = m * 16 + lane_id / 4;
            float* p = stage_base(b, m);
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n, p += 4)
            {
                if (r0 < size_m)     { p[0] = frag_c[m][n][0]; p[1] = frag_c[m][n][1]; }
                if (r0 + 8 < size_m) { p[2] = frag_c[m][n][2]; p[3] = frag_c[m][n][3]; }
            }
        }
    };

    auto add_stage = [&] (int b)
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int r0 = m * 16 + lane_id / 4;
            const float* p = stage_base(b, m);
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n, p += 4)
            {
                if (r0 < size_m)     { frag_c[m][n][0] += p[0]; frag_c[m][n][1] += p[1]; }
                if (r0 + 8 < size_m) { frag_c[m][n][2] += p[2]; frag_c[m][n][3] += p[3]; }
            }
        }
    };

    // The CTA covering unit i of the flattened (n-tile, k-tile) space, i.e. the inverse of
    // slice_beg(b) = tiles_k * tiles_n * b / num_slices
    auto block_of_unit = [&] (int i) -> int
    {
        return (int) (((long long) (i + 1) * num_slices - 1) / (long long) (tiles_k * tiles_n));
    };

    // Output reduction
    auto reduce = [&] ()
    {
        #if EXL3_GEMM_H_ACC
            // Fold the fp16 MMA accumulators into the fp32 accumulators once per k-slice
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
            {
                float2 f0 = __half22float2(frag_c_h[m][n][0]);
                float2 f1 = __half22float2(frag_c_h[m][n][1]);
                frag_c[m][n][0] += f0.x; frag_c[m][n][1] += f0.y;
                frag_c[m][n][2] += f1.x; frag_c[m][n][3] += f1.y;
            }
        #endif

        // First reduce all partial sums along k for the current slice
        threadblock_reduce();

        // Process (partial) slices within column in reverse order so the threadblock doing the bottom slice is
        // free to proceed to the next column right away
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        // Stream-K parallel fixup. The chain below combines partials one CTA at a time, each hop a
        // global write, an acquire-spin and a read back, L = num_slices / tiles_n deep. Every
        // contributor here instead stages its partial into its own arena slot and leaves; the CTA
        // covering k-tile 0 -- which is `last`, and the lowest blockIdx.x in the column -- waits
        // ONCE and folds the rest. Same traffic, two round trips of latency instead of L.
        //
        // The fold order is a pure function of the partition, not of arrival: own partial, then
        // descending blockIdx.x, which is descending k. It is NOT the chain's association (the
        // chain folds left in descending k and rounds the running sum to C's dtype at every hop),
        // so this changes the bits. It is deterministic either way, which is the property the
        // shape pin rests on
        bool fixed_up = false;
        if constexpr (fixup_shape_ok)
        if (use_fixup && !(first && last))
        {
            if (!last)
            {
                if (!sub_k) write_stage((int) blockIdx.x);
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    asm volatile ("fence.acq_rel.gpu;\n");
                    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(lock_d));
                }
                clear_frag_c();
                return;
            }
            else
            {
                // Everyone above k-tile 0 has contributed exactly tiles_k - lock_d k-tiles
                barrier_acquire(lock, tiles_k - lock_d);
                if (!sub_k)
                    for (int b = block_of_unit((slice2_n + 1) * tiles_k - 1);
                         b > (int) blockIdx.x; --b)
                        add_stage(b);
                fixed_up = true;
            }
        }

        if (!fixed_up)
            barrier_acquire(lock, lock_i);

        // Second and subsequent threadblocks in column read back the intermediate sum from global memory
        if (!sub_k && !first && !fixed_up)
        {
            read_sum_gl();
        }

        // All but last threadblock in column write the intermediate result to global memory
        if (!sub_k && !last)
        {
            write_sum_gl();
        }

        // Last block writes in row-major format
        if (!sub_k && last)
        {
            if constexpr (!shmem_out_had)
                write_sum_gl();
        }

        if constexpr (shmem_out_had)
        {
            // sh_c stages one 16-row block at a time and is reused by the next block, so each
            // block's transform must complete before the next one overwrites the staging area
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                if (m && last) __syncthreads();
                if (!sub_k && last) write_sum_tile_sh(m);
                if (last) __syncthreads();
                if (!sub_k && last) output_had_sh_gl(m);
            }
        }

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    // Wait until there are at most SH_STAGES - 2 async copies pending, i.e. at least one stage has finished loading
    auto wait_stage = [&] ()
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    // Perform tensor core matmul on current tile
    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                #if EXL3_GEMM_H_ACC
                    ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c_h[m][n]);
                #else
                    ptx_mma_m16n8k16(frag_a[buf][m], frag_b[buf][n], frag_c[m][n]);
                #endif
            }
        }
    };

    // Start global to shared pipeline
    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();
    wait_stage();

    // Start shared to register pipeline.
    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    // Main loop. Fragments are double buffered to allow more interleaving. This is especially important to hide the
    // dequantization overhead, but we need two different iterations of the main loop to avoid confusing the compiler
    // and making it (sometimes) place the fragment arrays in local memory

    #define FSTAGE_OLD(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        async_load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1)
    {
        while (true)
        {
            FSTAGE_OLD(0, 0);
        }
    }

    if constexpr (FRAG_STAGES == 2)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(0, 1);
        }
    }

    if constexpr (FRAG_STAGES == 3)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(0, 2);
        }
    }

    if constexpr (FRAG_STAGES == 4)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(0, 3);
        }
    }

    if constexpr (FRAG_STAGES == 5)
    {
        while (true)
        {
            FSTAGE(1, 0);
            FSTAGE(2, 1);
            FSTAGE(3, 2);
            FSTAGE(4, 3);
            FSTAGE(0, 4);
        }
    }
}

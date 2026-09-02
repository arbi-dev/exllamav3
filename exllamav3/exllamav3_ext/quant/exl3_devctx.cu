#include <cstdlib>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "exl3_devctx.cuh"
#include "../util.h"
#include "../util.cuh"

//DevCtx::DevCtc()
//{
//    int num_sms[MAX_DEVICES] = {};
//    int cc[MAX_DEVICES] = {};
//    void* locks[MAX_DEVICES] = {};
//    std::mutex mtx;
//}

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
    }
    return cc[device];
}

void* DevCtx::get_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device])
    {
        cudaSetDevice(device);
        cudaMalloc(&ws[device], WORKSPACE_SIZE);
    }
    return ws[device];
}

bool exl3_gemm_parallel_fixup_env()
{
    static const bool cached = []
    {
        const char* e = getenv("EXL3_GEMM_PARALLEL_FIXUP");
        return e && e[0] && e[0] != '0';
    }();
    return cached;
}

int* DevCtx::get_locks(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device])
    {
        cudaSetDevice(device);
        // The stream-K fixup arena is allocated only when the flag is on. Sized from a compile-time
        // grid bound rather than grown on demand: the buffer is allocated once and a captured graph
        // bakes the pointer, so growing it would either dangle or make the set of shapes that take
        // the fixup depend on the order shapes were first seen -- a boot-order-keyed reassociation
        size_t ints = MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS + MGEMM_SLOTS_INTS;
        if (exl3_gemm_parallel_fixup_env()) ints += EXL3_GEMM_FIXUP_INTS;
        size_t size = ints * sizeof(int);
        cudaMalloc(&locks[device], size);
        cudaMemset(locks[device], 0, size);
        fixup_arena_live[device] = exl3_gemm_parallel_fixup_env();
        if (fixup_arena_live[device])
        {
            int one = 1;
            cudaMemcpy((int*) locks[device] + EXL3_GEMM_FIXUP_ENABLE_OFFSET, &one, sizeof(int),
                       cudaMemcpyHostToDevice);
        }
    }
    return (int*) locks[device];
}

bool DevCtx::gemm_fixup_arena(int device)
{
    get_locks(device);
    std::lock_guard<std::mutex> lock(mtx);
    return fixup_arena_live[device];
}

// FLIPS the already-allocated arena's enable word. The two reductions associate differently, so a
// process that flips this mid-stream returns different bits for the same call -- exactly the
// dependence the shape pin exists to remove. It exists so an A/B can round-robin both reductions
// in ONE process instead of comparing across two, and it REFUSES when the arena was never
// allocated, so a caller cannot turn the fixup on by this door alone
void DevCtx::set_gemm_parallel_fixup(int device, int enable)
{
    int* base = get_locks(device);
    std::lock_guard<std::mutex> lock(mtx);
    if (!fixup_arena_live[device])
    {
        TORCH_CHECK(!enable,
            "exl3 parallel fixup: no arena on device ", device,
            ". The arena is allocated only when EXL3_GEMM_PARALLEL_FIXUP is set in the "
            "environment before the first GEMM; this switch can only flip an arena that exists.");
        return;
    }
    cudaMemcpy(base + EXL3_GEMM_FIXUP_ENABLE_OFFSET, &enable, sizeof(int), cudaMemcpyHostToDevice);
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_locks(device);
}

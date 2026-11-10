#pragma once

#include <tuple>
#include <mutex>
#include "../arch.cuh"

// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1] retired groups,
// [2 + group] ticket published to group. Self-resetting, zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)

// mgemm active-slot list for expert-range filtering, after the scheduler state: [0] total count,
// [1 + c] count for compaction chunk c, then the slot positions, in order. Upstream stages the
// filter in a fixed 128-entry __device__ array with no bound check, which overruns once a
// filtered launch carries more slots than that; this list is sized and offset instead
#define MGEMM_MAX_SLOTS 16384
#define MGEMM_CHUNKS 32
#define MGEMM_SLOTS_OFFSET (MOE_SCHED_OFFSET + MOE_SCHED_INTS)
#define MGEMM_SLOTS_INTS (1 + MGEMM_CHUNKS + MGEMM_MAX_SLOTS)

// Process-wide kernel switches, after the mgemm slot list, written once when the lock buffer is
// allocated. [0]: exl3_mgemm synchronizes per z-group (group_barrier) instead of the whole grid
// below sm_90 too (EXL3_MGEMM_GROUP_BARRIER=1). Both are correct under the cooperative launch,
// which makes every block co-resident; sm_90+ always uses the group barrier
#define MGEMM_FLAGS_OFFSET (MGEMM_SLOTS_OFFSET + MGEMM_SLOTS_INTS)
#define MGEMM_FLAGS_INTS 1

#if defined(USE_ROCM)
    // Whole-grid barrier of the EXL3 GEMM kernels (counter + sense), which ROCm launches without the
    // cooperative launch (see EXL3_COOP_LAUNCH in coop_autotune.cuh). After the mgemm slot list and
    // flags, which upstream does not have, so the two never alias
    #define EXL3_GRID_BARRIER_OFFSET (MGEMM_FLAGS_OFFSET + MGEMM_FLAGS_INTS)
    #define EXL3_GRID_BARRIER_INTS 2
#endif

// Workspace size
#define WORKSPACE_SIZE (16*1024*1024)

#define MAX_DEVICES 16
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  5

// Singleton to manage context for each device. Stores device attributes and a large-enough lock buffer per device
class DevCtx
{
private:
    int num_sms[MAX_DEVICES] = {};
    int cc[MAX_DEVICES] = {};
    int smem_max[MAX_DEVICES] = {};
    void* locks[MAX_DEVICES] = {};
    void* ws[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance();
    int get_num_sms(int device);
    int get_cc(int device);
    // Device capability: dynamic shared memory per block the driver will grant, unclamped
    // (sm_86 reports 99 KB, Turing 64 KB). Use this to decide what a device can do.
    int get_smem_max(int device);
    // What an EXL3 kernel may actually request: the above, capped at the SMEM_MAX the kernels
    // are written against. Use this for cudaFuncSetAttribute and launch parameters.
    int get_smem_request(int device);
    void* get_ws(int device);
    int* get_locks(int device);

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);
int g_get_smem_max(int device);

void prepare_ctx(int device);
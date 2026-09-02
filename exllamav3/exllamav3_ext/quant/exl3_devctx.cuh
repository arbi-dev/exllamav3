#pragma once

#include <tuple>
#include <mutex>

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
// [1 + c] count for compaction chunk c, then the slot positions, in order
#define MGEMM_MAX_SLOTS 16384
#define MGEMM_CHUNKS 32
#define MGEMM_SLOTS_OFFSET (MOE_SCHED_OFFSET + MOE_SCHED_INTS)
#define MGEMM_SLOTS_INTS (1 + MGEMM_CHUNKS + MGEMM_MAX_SLOTS)

// Stream-K parallel-fixup staging, after the mgemm slot list, in the same per-device buffer the
// locks live in so no kernel signature carries a second pointer. [0] is the enable word the host
// writes once from EXL3_GEMM_PARALLEL_FIXUP; the rest is the partial-tile arena.
//
// The arena is a fixed VRAM budget rather than a per-shape allocation because the buffer is
// allocated once per device and a captured graph bakes the pointer: a grow-on-demand arena would
// either dangle or make the eligible shape set depend on the order shapes were first seen, which
// is a boot-order-keyed numerics dependence. A shape whose staging does not fit the budget falls
// back to the ordered chain; the kernel evaluates that predicate itself from gridDim and its own
// tile constants, so host and device cannot disagree about which reduction ran.
#define EXL3_GEMM_FIXUP_ENABLE_OFFSET (MGEMM_SLOTS_OFFSET + MGEMM_SLOTS_INTS)
#define EXL3_GEMM_FIXUP_BYTES (16*1024*1024)
// Compile-time bound on gridDim.x, so a shape whose staging cannot fit the arena at ANY grid
// width compiles the staging path away entirely instead of carrying its register pressure into
// a kernel that will never take it. Checked again at runtime against the real grid
#define EXL3_GEMM_FIXUP_MAX_SLICES 256
#define EXL3_GEMM_FIXUP_FLOATS (EXL3_GEMM_FIXUP_BYTES / 4)
// Rounded to a 16-byte boundary: the staging stores two adjacent floats per fragment element and
// the compiler is free to merge them into one 8-byte store, which faults on a 4-byte-aligned base
#define EXL3_GEMM_FIXUP_OFFSET (((EXL3_GEMM_FIXUP_ENABLE_OFFSET + 1) + 3) & ~3)
#define EXL3_GEMM_FIXUP_INTS (4 + EXL3_GEMM_FIXUP_FLOATS)

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
    void* locks[MAX_DEVICES] = {};
    void* ws[MAX_DEVICES] = {};
    bool fixup_arena_live[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance();
    int get_num_sms(int device);
    int get_cc(int device);
    void* get_ws(int device);
    int* get_locks(int device);
    void set_gemm_parallel_fixup(int device, int enable);
    bool gemm_fixup_arena(int device);

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

int g_get_cc(int device);
int g_get_num_sms(int device);

// EXL3_GEMM_PARALLEL_FIXUP, read once per process. It decides whether the arena is ALLOCATED,
// so a build that leaves the flag off pays no VRAM for it at all
bool exl3_gemm_parallel_fixup_env();

void prepare_ctx(int device);
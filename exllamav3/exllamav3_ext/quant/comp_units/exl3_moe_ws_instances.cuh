#pragma once

// Per-K compilation units of the exl3_moe_coop2 ws kernels (exl3_moe_ws_kernel.cuh): each
// exl3_moe_ws_inst_k{K}.cu instantiates kernels A (gate/up) and B (down) at K bits for the three
// codebooks and both run lengths (16 and 32 rows)

#define EXL3_MOE_WS_INST_DECL(K) \
    void* exl3_moe_ws_kernel_a_k##K(int cb, int rm); \
    void* exl3_moe_ws_kernel_b_k##K(int cb, int rm);

EXL3_MOE_WS_INST_DECL(1)
EXL3_MOE_WS_INST_DECL(2)
EXL3_MOE_WS_INST_DECL(3)
EXL3_MOE_WS_INST_DECL(4)
EXL3_MOE_WS_INST_DECL(5)
EXL3_MOE_WS_INST_DECL(6)
EXL3_MOE_WS_INST_DECL(7)
EXL3_MOE_WS_INST_DECL(8)

#undef EXL3_MOE_WS_INST_DECL

#define EXL3_MOE_WS_PICK(KERN, K, cb, rm) \
    (rm == 32 ? (cb == 0 ? (void*) KERN<K, 0, 32> : cb == 1 ? (void*) KERN<K, 1, 32> : (void*) KERN<K, 2, 32>) \
              : (cb == 0 ? (void*) KERN<K, 0, 16> : cb == 1 ? (void*) KERN<K, 1, 16> : (void*) KERN<K, 2, 16>))

#define EXL3_MOE_WS_INST_DEF(K) \
    void* exl3_moe_ws_kernel_a_k##K(int cb, int rm) \
    { \
        using namespace exl3_moe_ws_ns; \
        return EXL3_MOE_WS_PICK(exl3_moe_ws_a_kernel, K, cb, rm); \
    } \
    void* exl3_moe_ws_kernel_b_k##K(int cb, int rm) \
    { \
        using namespace exl3_moe_ws_ns; \
        return EXL3_MOE_WS_PICK(exl3_moe_ws_b_kernel, K, cb, rm); \
    }

#pragma once

// Per-K compilation units of the exl3_moe_coop2 ws kernels (exl3_moe_ws_kernel.cuh): each
// exl3_moe_ws_inst_k{K}.cu instantiates kernels A (gate/up) and B (down) with routed experts at K
// bits and a folded shared expert at K or 6 bits (Ks), for the three codebooks and both run
// lengths (16 and 32 rows). Ks is K when no shared expert is folded

#define EXL3_MOE_WS_INST_DECL(K) \
    void* exl3_moe_ws_kernel_a_k##K(int ks, int cb, int rm); \
    void* exl3_moe_ws_kernel_b_k##K(int ks, int cb, int rm);

EXL3_MOE_WS_INST_DECL(1)
EXL3_MOE_WS_INST_DECL(2)
EXL3_MOE_WS_INST_DECL(3)
EXL3_MOE_WS_INST_DECL(4)
EXL3_MOE_WS_INST_DECL(5)
EXL3_MOE_WS_INST_DECL(6)
EXL3_MOE_WS_INST_DECL(7)
EXL3_MOE_WS_INST_DECL(8)

#undef EXL3_MOE_WS_INST_DECL

#define EXL3_MOE_WS_PICK(KERN, K, KS, cb, rm) \
    (rm == 32 ? (cb == 0 ? (void*) KERN<K, KS, 0, 32> : cb == 1 ? (void*) KERN<K, KS, 1, 32> : (void*) KERN<K, KS, 2, 32>) \
              : (cb == 0 ? (void*) KERN<K, KS, 0, 16> : cb == 1 ? (void*) KERN<K, KS, 1, 16> : (void*) KERN<K, KS, 2, 16>))

#define EXL3_MOE_WS_INST_DEF(K) \
    void* exl3_moe_ws_kernel_a_k##K(int ks, int cb, int rm) \
    { \
        using namespace exl3_moe_ws_ns; \
        if (ks == K) return EXL3_MOE_WS_PICK(exl3_moe_ws_a_kernel, K, K, cb, rm); \
        if (ks == 6) return EXL3_MOE_WS_PICK(exl3_moe_ws_a_kernel, K, 6, cb, rm); \
        return nullptr; \
    } \
    void* exl3_moe_ws_kernel_b_k##K(int ks, int cb, int rm) \
    { \
        using namespace exl3_moe_ws_ns; \
        if (ks == K) return EXL3_MOE_WS_PICK(exl3_moe_ws_b_kernel, K, K, cb, rm); \
        if (ks == 6) return EXL3_MOE_WS_PICK(exl3_moe_ws_b_kernel, K, 6, cb, rm); \
        return nullptr; \
    }

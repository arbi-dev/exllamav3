#pragma once

// Per-K compilation units of the exl3_moe_coop2 kernels (exl3_moe_coop2_kernel.cuh): each
// exl3_moe_coop2_inst_k{K}.cu instantiates kernels A (gate/up) and B (down) for the three codebooks

#define EXL3_MOE_COOP2_INST_DECL(K) \
    void* exl3_moe_coop2_kernel_a_k##K(int cb); \
    void* exl3_moe_coop2_kernel_b_k##K(int cb);

EXL3_MOE_COOP2_INST_DECL(1)
EXL3_MOE_COOP2_INST_DECL(2)
EXL3_MOE_COOP2_INST_DECL(3)
EXL3_MOE_COOP2_INST_DECL(4)
EXL3_MOE_COOP2_INST_DECL(5)
EXL3_MOE_COOP2_INST_DECL(6)
EXL3_MOE_COOP2_INST_DECL(7)
EXL3_MOE_COOP2_INST_DECL(8)

#undef EXL3_MOE_COOP2_INST_DECL

#define EXL3_MOE_COOP2_INST_DEF(K) \
    void* exl3_moe_coop2_kernel_a_k##K(int cb) \
    { \
        using namespace exl3_moe_coop2_ns; \
        switch (cb) \
        { \
            case 0: return (void*) exl3_moe_coop2_a_kernel<K, 0>; \
            case 1: return (void*) exl3_moe_coop2_a_kernel<K, 1>; \
            default: return (void*) exl3_moe_coop2_a_kernel<K, 2>; \
        } \
    } \
    void* exl3_moe_coop2_kernel_b_k##K(int cb) \
    { \
        using namespace exl3_moe_coop2_ns; \
        switch (cb) \
        { \
            case 0: return (void*) exl3_moe_coop2_b_kernel<K, 0>; \
            case 1: return (void*) exl3_moe_coop2_b_kernel<K, 1>; \
            default: return (void*) exl3_moe_coop2_b_kernel<K, 2>; \
        } \
    }

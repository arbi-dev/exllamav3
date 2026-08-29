#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// acc_mode selects the cuBLAS COMPUTE TYPE for this call, not for the process.
// 0 = CUBLAS_COMPUTE_32F (fp32 accumulate), 1 = CUBLAS_COMPUTE_16F (fp16
// accumulate, ~1.8x the rate on Ada at ~10x the relative error). It is per-call
// because hgemm_gr is shared: the trellis reconstruct leg wants the fast
// accumulator, while MoE routing, the attention gates and the MLA index GEMMs
// are small, ill-conditioned or both and must keep fp32. cuBLAS plans per
// (shape, compute type), so a caller must pass the SAME mode during warmup and
// capture as it does at serve time or the captured plan is the other mode's.
void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph,
    int acc_mode = 0
);

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int acc_mode = 0
);
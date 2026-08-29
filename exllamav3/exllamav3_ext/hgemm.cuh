#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

// Accumulator width for hgemm's fp16-OUTPUT path.
//
// cuBLAS reads the accumulator out of the compute type, and on the GeForce
// tensor cores (Ampere/Ada consumer parts) fp16 inputs with an fp32
// accumulator run at HALF the rate of fp16 inputs with an fp16 accumulator.
// The fp32-output path has no choice: cublasGemmEx has no fp16-accumulate
// compute type that writes an fp32 C, so HGEMM_ACCUM_FP16 is ignored there.
enum HGemmAccum : int
{
    // Resolve from the EXL3_HGEMM_FP16_ACCUM environment variable. Honoured
    // ONLY at the Python binding (see bindings.cpp); the C++ entry points
    // below default to HGEMM_ACCUM_FP32 so that an env var set for the
    // reconstruct+hgemm prefill leg cannot silently move the accumulator of
    // an unrelated in-kernel hgemm (MoE router scores, attention gates, the
    // DSA/MLA index GEMMs).
    HGEMM_ACCUM_DEFAULT = -1,

    // CUBLAS_COMPUTE_32F.
    HGEMM_ACCUM_FP32 = 0,

    // CUBLAS_COMPUTE_16F.
    HGEMM_ACCUM_FP16 = 1,

    // CUBLAS_COMPUTE_32F_FAST_16F. The CUDA header describes this as "float -
    // fast, allows down-converting inputs to half or TF32"
    // (cublas_api.h, cublasComputeType_t): it relaxes the INPUT width and
    // leaves the accumulator at fp32. Both inputs here are already fp16, so
    // there is nothing left to down-convert and this is expected to be
    // indistinguishable from HGEMM_ACCUM_FP32 in both speed and error. It is
    // exposed so that expectation can be falsified by an A/B rather than
    // argued.
    HGEMM_ACCUM_FP32_FAST_16F = 2,
};

// The mode HGEMM_ACCUM_DEFAULT resolves to. Reads EXL3_HGEMM_FP16_ACCUM once,
// on first call, and caches it (function-local static, so the getenv never
// lands on a hot path). Unset / empty / "0" -> HGEMM_ACCUM_FP32.
int hgemm_default_accum();

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph,
    int accum = HGEMM_ACCUM_FP32
);

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int accum = HGEMM_ACCUM_FP32
);

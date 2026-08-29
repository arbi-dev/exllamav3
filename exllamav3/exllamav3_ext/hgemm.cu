#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#include <cstdlib>
#include <cstring>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
- if c is float16, operation is float16 @ float16 -> float16, and the ACCUMULATOR
  is selected by the `accum` argument (see HGemmAccum in hgemm.cuh); it is fp32
  unless the caller asks otherwise.
*/

using bfloat16 = __nv_bfloat16;

int hgemm_default_accum()
{
    static const int mode = []() -> int
    {
        const char* v = std::getenv("EXL3_HGEMM_FP16_ACCUM");
        if (!v || !*v) return HGEMM_ACCUM_FP32;
        char* end = nullptr;
        long parsed = std::strtol(v, &end, 10);
        if (end == v) return HGEMM_ACCUM_FP32;
        switch (parsed)
        {
            case HGEMM_ACCUM_FP16: return HGEMM_ACCUM_FP16;
            case HGEMM_ACCUM_FP32_FAST_16F: return HGEMM_ACCUM_FP32_FAST_16F;
            default: return HGEMM_ACCUM_FP32;
        }
    }();
    return mode;
}

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream,
    int accum
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    // An fp16 accumulator only exists for an fp16 C: cublasGemmEx has no
    // compute type that accumulates in half and stores float.
    bool fp16_accum = output_fp16 && accum == HGEMM_ACCUM_FP16;

    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    cublasStatus_t r;

    if (fp16_accum)
    {
        // alpha/beta are read as the COMPUTE type, so CUBLAS_COMPUTE_16F wants
        // half scalars — the float pair below would be misread as two halves.
        // (This is the pair cublasHgemm took before 69c8ee8 folded the fp16
        // path into cublasGemmEx.)
        half alpha_ = __float2half(1.0f);
        half beta_ = __float2half(0.0f);
        r = cublasGemmEx
        (
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, CUDA_R_16F, size_n,
                     a_ptr, CUDA_R_16F, size_k,
            &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    else
    {
        float alpha_ = 1.0f;
        float beta_ = 0.0f;
        cublasComputeType_t compute_type =
            accum == HGEMM_ACCUM_FP32_FAST_16F ? CUBLAS_COMPUTE_32F_FAST_16F
                                               : CUBLAS_COMPUTE_32F;
        r = cublasGemmEx
        (
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, CUDA_R_16F, size_n,
                     a_ptr, CUDA_R_16F, size_k,
            &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
            compute_type,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph,
    int accum
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream, accum);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int accum
)
{
    hgemm_gr(a, b, c, nullptr, accum);
}

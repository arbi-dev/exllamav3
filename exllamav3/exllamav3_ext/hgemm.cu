#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- inputs are always float16; c may be float16 or float32
- acc_mode selects the ACCUMULATOR independently of c's dtype:
    0 -> CUBLAS_COMPUTE_32F  (fp32 accumulate)
    1 -> CUBLAS_COMPUTE_16F  (fp16 accumulate; requires an fp16 c)
*/

using bfloat16 = __nv_bfloat16;

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream,
    int acc_mode
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

    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;

    // alpha/beta must be in the COMPUTE type, not always float: cuBLAS reads
    // them through a pointer typed by the compute type, so handing a float* to
    // a 16F compute reads two halves out of one float and scales by garbage.
    TORCH_CHECK(acc_mode == 0 || acc_mode == 1, "acc_mode must be 0 (fp32) or 1 (fp16)");
    TORCH_CHECK(acc_mode == 0 || output_fp16, "acc_mode 1 requires a float16 c");
    cublasComputeType_t compute = acc_mode ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
    float alpha_f = 1.0f, beta_f = 0.0f;
    half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    const void* alpha_ = acc_mode ? (const void*) &alpha_h : (const void*) &alpha_f;
    const void* beta_ = acc_mode ? (const void*) &beta_h : (const void*) &beta_f;

    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        alpha_, b_ptr, CUDA_R_16F, size_n,
                a_ptr, CUDA_R_16F, size_k,
        beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        compute,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph,
    int acc_mode
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream, acc_mode);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int acc_mode
)
{
    hgemm_gr(a, b, c, nullptr, acc_mode);
}

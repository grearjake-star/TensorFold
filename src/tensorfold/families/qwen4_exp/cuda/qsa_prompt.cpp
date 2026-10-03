#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

void qsa_scores_cuda(const __nv_bfloat16*, const __nv_bfloat16*, const int*, float*, int*, int, int, int, int, int,
                     cudaStream_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == t, name, ": expected a contiguous CUDA tensor");
}

// iq [R, 4, 128] bf16, pooled [*, 128] bf16, pos0 [1] int32, scores [>= R, NB] fp32, smx [>= R, NSUB] int32 or empty
void scores(const at::Tensor& iq, const at::Tensor& pooled, const at::Tensor& pos0, at::Tensor sc, at::Tensor smx,
            int64_t rows, int64_t blocks, int64_t rt) {
    check(iq, at::kBFloat16, "iq");
    check(pooled, at::kBFloat16, "pooled");
    check(pos0, at::kInt, "pos0");
    check(sc, at::kFloat, "scores");
    TORCH_CHECK(iq.dim() == 3 && iq.size(1) == 4 && iq.size(2) == 128 && iq.size(0) >= rows,
                "the prompt indexer kernel is built for 4 heads of 128");
    TORCH_CHECK(pooled.dim() == 2 && pooled.size(1) == 128 && pooled.size(0) >= blocks, "pooled keys [blocks, 128]");
    TORCH_CHECK(sc.dim() == 2 && sc.size(0) >= rows && sc.size(1) >= blocks, "scores [rows, NB]");
    int* sp = nullptr;
    int64_t nsub = 0;
    if (smx.numel()) {
        check(smx, at::kInt, "smx");
        TORCH_CHECK(smx.dim() == 2 && smx.size(0) >= rows && smx.size(1) * 8 >= blocks, "smx [rows, NB / 8]");
        sp = smx.data_ptr<int>();
        nsub = smx.size(1);
    }
    if (rows <= 0 || blocks <= 0) return;
    c10::cuda::CUDAGuard guard(iq.device());
    qsa_scores_cuda(reinterpret_cast<const __nv_bfloat16*>(iq.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16*>(pooled.data_ptr()), pos0.data_ptr<int>(),
                    sc.data_ptr<float>(), sp, (int)rows, (int)sc.size(1), (int)nsub, (int)blocks, (int)rt,
                    c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scores", &scores); }

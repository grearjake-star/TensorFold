// EXL3 routed experts, any codebook and a width per expert: fixed-order splits, slots and butterflies, no atomics; 4-bit mcg matches GLM's kernel bit for bit.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mutex>
#include <unordered_map>

#include "experts_prompt.cuh"

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Grouping in one block: distinct experts (< E) in id order, members row * 32 + slot in row order, -1 after the last.
constexpr int GROUP_THREADS = 1024;
constexpr int GROUP_PER_THREAD = 4;

__global__ void __launch_bounds__(GROUP_THREADS) group_kernel(const int* __restrict__ pick, int* __restrict__ uids,
                                                              int* __restrict__ ucount, int* __restrict__ members,
                                                              int R, int slots, int E, int maxm) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[GROUP_THREADS / 32];
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += GROUP_THREADS) sh_pick[i] = pick[i];
    __syncthreads();
    int cnt[GROUP_PER_THREAD];
    int used = 0;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        int c = 0;
        if (e < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == e;
        cnt[q] = c;
        used += c > 0;
    }
    // exclusive scan of `used` over threads
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = warp_tot[lane];
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int x = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += x;
        }
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < GROUP_PER_THREAD; ++q) {
        if (cnt[q] == 0) continue;
        const int e = threadIdx.x * GROUP_PER_THREAD + q;
        uids[place] = e;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == e) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
}

// Walsh-Hadamard transform of 128 values, 4 a lane, fixed butterfly order (strides 1, 2 in registers, 4..64 across lanes).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }

// Program (member row, 128-block of K, matrix): Xh = fp16((x * suh) @ H) for gate and up of every routed slot (pick < E).
template <typename TIN>
__global__ void rot_in_kernel(const TIN* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// rot_in_kernel's arithmetic per warp, four 128-blocks of K a program (128 threads): fewer, fuller blocks (W5-6).
template <typename TIN>
__global__ void __launch_bounds__(128) rot_in4_kernel(const TIN* __restrict__ x, int x_stride,
                                                      const int* __restrict__ pick, const half* __restrict__ suh0,
                                                      const half* __restrict__ suh1, half* __restrict__ out0,
                                                      half* __restrict__ out1, int K, int slots, int E) {
    const int p = blockIdx.x, blk = blockIdx.y * 4 + (threadIdx.x >> 5), mat = blockIdx.z;
    const int row = p / slots;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x & 31;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the width): splits summed in order, rotated, * svh, SwiGLU (0: GLM's bf16 roundings, 1: fp32), then Xd = fp16((act * suh_d) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int E, float limit, int act_mode) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float act;
        if (act_mode == 0) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {
            float gg = fminf(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]), limit);
            float uu = fminf(fmaxf(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]), -limit), limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): Y = (splits summed in order) @ H * svh_d, fp32.
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int E) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int e = pick[p];
    if (e < 0 || e >= E) return;
    const int lane = threadIdx.x;
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// out[r][d] = sum over slots in order of wts[r][k] * y[r * slots + k][d] (fp32, fma chain from 0).
__global__ void combine_kernel(const float* __restrict__ y, const float* __restrict__ wts, float* __restrict__ out,
                               int D, int slots) {
    const int r = blockIdx.x;
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = fmaf(wts[r * slots + k], y[((size_t)r * slots + k) * D + d], acc);
    out[(size_t)r * D + d] = acc;
}

// down_epilogue_kernel then combine_kernel in one launch, the same arithmetic in the same order (the same bits).
__global__ void down_combine_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                    const half* __restrict__ svh_d, float* __restrict__ y,
                                    const float* __restrict__ wts, float* __restrict__ out, int P, int D, int SK,
                                    int E, int slots) {
    __shared__ float4 part[32][32];                 // [slot][lane]: the slot's 4 outputs of the lane
    const int r = blockIdx.x, blk = blockIdx.y;
    const int k = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blk * 128 + 4 * lane;
    const int p = r * slots + k;
    const int e = pick[p];
    float o[4];
    if (e >= 0 && e < E) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
            for (int q = 0; q < SK; ++q) s += Z[((size_t)q * P + p) * D + n + j];
            v[j] = s;
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
            y[(size_t)p * D + n + j] = o[j];
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = y[(size_t)p * D + n + j];
    }
    part[k][lane] = make_float4(o[0], o[1], o[2], o[3]);
    __syncthreads();
    if (k != 0) return;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int q = 0; q < slots; ++q) {
        const float w = wts[r * slots + q];
        const float4 u = part[q][lane];
        acc[0] = fmaf(w, u.x, acc[0]);
        acc[1] = fmaf(w, u.y, acc[1]);
        acc[2] = fmaf(w, u.z, acc[2]);
        acc[3] = fmaf(w, u.w, acc[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) out[(size_t)r * D + n + j] = acc[j];
}

// ---- S1-ROUTE (TF_MOE_ROUTE_FUSED): tensorfold.cuda.moe._topk_rows + group_kernel + rot_in_kernel in one launch ----
// Same bits as the three launches: the top-k reproduces the Triton kernel's PTX (max.f32 reductions over the same
// next_pow2(NE + 1) values padded with -inf, the lowest id among equal maxima, ex = ex2.approx.f32((m - top) * log2e),
// total = 0 + ex_0 + ex_1 + ... in pick order, w = bf16(div.full.f32(ex, total)), the shared slot NE with weight
// bf16(div.full.f32(1, ex2.approx.f32((0 - bf16(l_NE)) * log2e) + 1))); rot_in is rot_in_kernel's per-lane arithmetic
// (written out the same way); the grouping is integers (group_kernel's order: experts by id, members in pair order).
// Grid (R * slots, K / 512, 2), 128 threads: block (pair p, 4 blocks of 128 of K, gate/up) takes its own pick with
// k + 1 rounds of a warp top-k and rotates x into xg/xu; the row's slot-0 block (y = z = 0) also writes the row's picks
// and weights; the last of those R blocks to finish (a counter, reset by it) groups every pick.
constexpr float ROUTE_LOG2E = 1.44269502162933349609375f;   // 0x3FB8AA3B: Triton's fp32 tl.exp scale

__device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float div_full_f32(float a, float b) {
    float y;
    asm volatile("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// One warp: `rounds` picks of row L (NE logits; NV * 32 = next_pow2(NE + 1) values, the rest -inf). want_idx = the id
// of round `want`; lane k (< rounds) gets round k's id and ex; total = the sum of the rounds' ex in order.
template <int NV>
__device__ __forceinline__ void warp_topk(const float* __restrict__ L, int NE, int rounds, int want, int lane,
                                          int& want_idx, int& my_id, float& my_ex, float& total) {
    float v[NV];
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        const int i = lane + 32 * j;
        v[j] = i < NE ? L[i] : -INFINITY;
    }
    float top = v[0];
#pragma unroll
    for (int j = 1; j < NV; ++j) top = fmaxf(top, v[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, o));
    total = 0.f;
    for (int k = 0; k < rounds; ++k) {
        float m = v[0];
#pragma unroll
        for (int j = 1; j < NV; ++j) m = fmaxf(m, v[j]);
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        int c = NV * 32;
#pragma unroll
        for (int j = NV - 1; j >= 0; --j)
            if (v[j] == m) c = lane + 32 * j;
#pragma unroll
        for (int o = 16; o; o >>= 1) c = min(c, __shfl_xor_sync(0xffffffffu, c, o));
        const float ex = ex2_approx_f32(__fmul_rn(__fsub_rn(m, top), ROUTE_LOG2E));
        total = __fadd_rn(total, ex);
        if (lane == k) {
            my_id = c;
            my_ex = ex;
        }
        if (k == want) want_idx = c;
#pragma unroll
        for (int j = 0; j < NV; ++j)
            if (lane + 32 * j == c) v[j] = -INFINITY;
    }
}

// rot_in_kernel's lane arithmetic, the same expressions in the same order.
template <typename TIN>
__device__ __forceinline__ void route_rot_lane(const TIN* __restrict__ x, int x_stride, int row, int p, int e, int blk,
                                               int mat, const half* __restrict__ suh0, const half* __restrict__ suh1,
                                               half* __restrict__ out0, half* __restrict__ out1, int K, int lane) {
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const TIN* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = to_f<TIN>(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

constexpr int ROUTE_THREADS = 128;
constexpr int ROUTE_GPT = 8;                 // experts a thread counts while grouping: E <= 1024

template <typename TIN, int NV>
__global__ void __launch_bounds__(ROUTE_THREADS) route_kernel(
    const float* __restrict__ L, int NL, int NE, int topk, int* __restrict__ pick, float* __restrict__ wts,
    const TIN* __restrict__ x, int x_stride, const half* __restrict__ suh0, const half* __restrict__ suh1,
    half* __restrict__ out0, half* __restrict__ out1, int K, int slots, int E, int* __restrict__ uids,
    int* __restrict__ ucount, int* __restrict__ members, int maxm, int R, int* __restrict__ counter) {
    extern __shared__ int sh_pick[];
    __shared__ int warp_tot[32];
    __shared__ int is_last;
    const int p = blockIdx.x, row = p / slots, k = p % slots;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool writer = k == 0 && blockIdx.y == 0 && blockIdx.z == 0;
    int e = NE;                                              // slot topk: the shared expert
    if (k < topk) {
        const bool full = writer && warp == 0;
        int idx = NV * 32, my_id = 0;
        float my_ex = 0.f, total = 0.f;
        warp_topk<NV>(L + (size_t)row * NL, NE, full ? topk : k + 1, k, lane, idx, my_id, my_ex, total);
        e = idx;
        if (full) {
            if (lane < topk) {
                pick[row * slots + lane] = my_id;
                wts[row * slots + lane] = bf16_round(div_full_f32(my_ex, total));
            } else if (lane == topk) {
                const float sg = bf16_round(L[(size_t)row * NL + NE]);
                const float den = __fadd_rn(ex2_approx_f32(__fmul_rn(__fsub_rn(0.f, sg), ROUTE_LOG2E)), 1.f);
                pick[row * slots + lane] = NE;
                wts[row * slots + lane] = bf16_round(div_full_f32(1.f, den));
            }
            __threadfence();
        }
    }
    if (e >= 0 && e < E) route_rot_lane<TIN>(x, x_stride, row, p, e, blockIdx.y * 4 + warp, blockIdx.z, suh0, suh1,
                                             out0, out1, K, lane);
    if (!writer) return;
    __syncthreads();
    if (threadIdx.x == 0) is_last = atomicAdd(counter, 1) == R - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    // group_kernel on 128 threads: the same outputs (integers)
    const int n = R * slots;
    for (int i = threadIdx.x; i < n; i += ROUTE_THREADS) sh_pick[i] = __ldcg(pick + i);
    __syncthreads();
    const int per = (E + ROUTE_THREADS - 1) / ROUTE_THREADS;
    int cnt[ROUTE_GPT];
    int used = 0;
#pragma unroll
    for (int q = 0; q < ROUTE_GPT; ++q) {
        const int ex = threadIdx.x * per + q;
        int c = 0;
        if (q < per && ex < E)
            for (int i = 0; i < n; ++i) c += sh_pick[i] == ex;
        cnt[q] = c;
        used += c > 0;
    }
    int inc = used;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += v;
    }
    if (lane == 31) warp_tot[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int v = lane < ROUTE_THREADS / 32 ? warp_tot[lane] : 0;
        int s = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        __syncwarp();
        warp_tot[lane] = s - v;                                   // exclusive per warp
        if (lane == 31) ucount[0] = s;
    }
    __syncthreads();
    int place = warp_tot[warp] + inc - used;
#pragma unroll
    for (int q = 0; q < ROUTE_GPT; ++q) {
        if (cnt[q] == 0) continue;
        const int ex = threadIdx.x * per + q;
        uids[place] = ex;
        int j = 0;
        for (int i = 0; i < n && j < maxm; ++i)
            if (sh_pick[i] == ex) members[place * maxm + j++] = (i / slots) * 32 + (i % slots);
        for (; j < maxm; ++j) members[place * maxm + j] = -1;
        ++place;
    }
    if (threadIdx.x == 0) *counter = 0;                       // ready for the next launch (stream order)
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------

namespace tf_exl3x {
extern template void grouped_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void grouped_items_launch<0>(const GroupedArgs&, cudaStream_t);
extern template void grouped_items_launch<1>(const GroupedArgs&, cudaStream_t);
extern template void grouped_items_launch<2>(const GroupedArgs&, cudaStream_t);
extern template void prompt_launch<0>(const PromptArgs&, int, cudaStream_t);
extern template void prompt_down_launch<0>(const DownArgs&, int, cudaStream_t);
extern template void prompt_launch<1>(const PromptArgs&, int, cudaStream_t);
extern template void prompt_down_launch<1>(const DownArgs&, int, cudaStream_t);
extern template void prompt_launch<2>(const PromptArgs&, int, cudaStream_t);
extern template void prompt_down_launch<2>(const DownArgs&, int, cudaStream_t);
extern template void prompt_diag_launch<2>(const PromptArgs&, int, cudaStream_t);
extern template void dequant_launch<0>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<1>(const uint32_t*, half*, int, int, int, cudaStream_t);
extern template void dequant_launch<2>(const uint32_t*, half*, int, int, int, cudaStream_t);
}  // namespace tf_exl3x

void exl3x_grouped_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                        const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& uids, const at::Tensor& ucount,
                        const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                        int64_t SK, int64_t slots, int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t lo,
                        int64_t hi) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = uids.data_ptr<int>();
    a.ucount = ucount.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = (int)members.size(1); a.slots = (int)slots;
    a.nexp_max = (int)uids.size(0);
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_grouped_items_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0,
                              const at::Tensor& TP1, const at::Tensor& B0, const at::Tensor& B1,
                              const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& Z,
                              int64_t mats, int64_t K, int64_t N, int64_t P, int64_t SK, int64_t E, int64_t items_max,
                              int64_t cb, int64_t nt, int64_t warps, int64_t pf, int64_t mtb, int64_t lo, int64_t hi) {
    TORCH_CHECK(K % (16 * SK * warps) == 0 && N % (16 * nt) == 0, "K and N must split evenly");
    tf_exl3x::GroupedArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.uids = nullptr;
    a.ucount = nullptr;
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.SK = (int)SK; a.maxm = 0; a.slots = 0; a.nexp_max = 0;
    a.mats = (int)mats; a.nt = (int)nt; a.warps = (int)warps; a.pf = (int)pf; a.lo = (int)lo; a.hi = (int)hi;
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.items_max = (int)items_max;
    a.mtb = (int)mtb;
    a.E = (int)E;
    if (items_max < 1) return;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (cb == 0) tf_exl3x::grouped_items_launch<0>(a, stream);
    else if (cb == 1) tf_exl3x::grouped_items_launch<1>(a, stream);
    else if (cb == 2) tf_exl3x::grouped_items_launch<2>(a, stream);
    else TORCH_CHECK(false, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Prompt instances on a side stream (TF_EXL3_PROMPT_SIDE): ``widths`` carries the K2 instances to launch in its low 32
// bits and, in its high 32 bits, those to run on a side stream beside the rest (the shared expert's width: a few items
// that otherwise run alone after the routed experts' instance). Every pair is computed by one program of one instance,
// which writes only that pair's rows, so where an instance runs never changes a bit. The side stream forks from and
// joins back into the current stream with events (also inside a CUDA graph capture).
namespace {
struct SideStream {
    at::cuda::CUDAStream stream;
    cudaEvent_t fork, join;
};
std::mutex side_mutex;
SideStream* side_stream(int device, cudaStream_t current) {
    static std::unordered_map<int, SideStream*> streams;
    auto it = streams.find(device);
    if (it != streams.end()) return it->second;
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(current, &capturing));
    if (capturing != cudaStreamCaptureStatusNone) return nullptr;     // no stream/event creation inside a capture
    auto* s = new SideStream{at::cuda::getStreamFromPool(true, static_cast<c10::DeviceIndex>(device)), nullptr, nullptr};
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&s->fork, cudaEventDisableTiming));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&s->join, cudaEventDisableTiming));
    streams[device] = s;
    return s;
}

// launch(mask, stream): the side widths first, on the side stream (high priority: its few programs start early),
// then the rest on ``stream``; the current stream waits for both.
template <typename Launch>
void with_side(int64_t widths, cudaStream_t stream, int device, Launch launch) {
    const int all = (int)(widths & 0xffffffffLL), side = (int)(widths >> 32) & all;
    if (side == 0 || side == all) {
        launch(all, stream);
        return;
    }
    std::lock_guard<std::mutex> lock(side_mutex);
    SideStream* sd = side_stream(device, stream);
    if (sd == nullptr) {                                  // first use inside a capture: one stream, the same bits
        launch(all, stream);
        return;
    }
    C10_CUDA_CHECK(cudaEventRecord(sd->fork, stream));
    C10_CUDA_CHECK(cudaStreamWaitEvent(sd->stream.stream(), sd->fork, 0));
    launch(side, sd->stream.stream());
    launch(all & ~side, stream);
    C10_CUDA_CHECK(cudaEventRecord(sd->join, sd->stream.stream()));
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, sd->join, 0));
}
}  // namespace

void exl3x_prompt_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& TP0, const at::Tensor& TP1,
                       const at::Tensor& B0, const at::Tensor& B1, const at::Tensor& items, const at::Tensor& counts,
                       const at::Tensor& members, at::Tensor& Z, int64_t mats, int64_t K, int64_t N, int64_t P,
                       int64_t E, int64_t seg, int64_t wps, int64_t items_max, int64_t cb, int64_t nt,
                       int64_t widths) {
    TORCH_CHECK(K % 16 == 0 && (K / 16) % (seg * wps) == 0 && N % (16 * nt) == 0, "prompt experts: K and N must split evenly");
    tf_exl3x::PromptArgs a;
    a.x0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.x1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.tp0 = TP0.data_ptr<int64_t>();
    a.tp1 = TP1.data_ptr<int64_t>();
    a.k2_0 = B0.data_ptr<int>();
    a.k2_1 = B1.data_ptr<int>();
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.z = Z.data_ptr<float>();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.E = (int)E; a.seg = (int)seg; a.wps = (int)wps;
    a.items_max = (int)items_max; a.mats = (int)mats; a.nt = (int)nt;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (widths < 0) {                                   // timing diagnostics (mul1, 4-bit only)
        TORCH_CHECK(cb == 2, "diagnostics: mul1 only");
        tf_exl3x::prompt_diag_launch<2>(a, (int)-widths, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    TORCH_CHECK(cb >= 0 && cb <= 2, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    with_side(widths, stream, X0.get_device(), [&](int w, cudaStream_t st) {
        if (cb == 0) tf_exl3x::prompt_launch<0>(a, w, st);
        else if (cb == 1) tf_exl3x::prompt_launch<1>(a, w, st);
        else tf_exl3x::prompt_launch<2>(a, w, st);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_prompt_down_cuda(const at::Tensor& X, const at::Tensor& TP, const at::Tensor& B, const at::Tensor& svh,
                            const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& Y,
                            int64_t K, int64_t N, int64_t P, int64_t E, int64_t seg, int64_t wps, int64_t items_max,
                            int64_t cb, int64_t widths) {
    TORCH_CHECK(K % 16 == 0 && (K / 16) % (seg * wps) == 0 && N % 128 == 0, "prompt down: K and N must split evenly");
    tf_exl3x::DownArgs a;
    a.x = reinterpret_cast<const half*>(X.data_ptr());
    a.tp = TP.data_ptr<int64_t>();
    a.k2 = B.data_ptr<int>();
    a.svh = reinterpret_cast<const half*>(svh.data_ptr());
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.y = Y.data_ptr();
    a.K = (int)K; a.N = (int)N; a.P = (int)P; a.E = (int)E; a.seg = (int)seg; a.wps = (int)wps;
    a.items_max = (int)items_max; a.bf16 = Y.scalar_type() == at::kBFloat16 ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(cb >= 0 && cb <= 2, "codebook must be 0 (3inst), 1 (mcg) or 2 (mul1)");
    with_side(widths, stream, X.get_device(), [&](int w, cudaStream_t st) {
        if (cb == 0) tf_exl3x::prompt_down_launch<0>(a, w, st);
        else if (cb == 1) tf_exl3x::prompt_down_launch<1>(a, w, st);
        else tf_exl3x::prompt_down_launch<2>(a, w, st);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_dequant_cuda(const at::Tensor& T, at::Tensor& out, int64_t K, int64_t N, int64_t k2, int64_t cb) {
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = reinterpret_cast<const uint32_t*>(T.data_ptr());
    auto o = reinterpret_cast<half*>(out.data_ptr());
    if (cb == 0) tf_exl3x::dequant_launch<0>(t, o, (int)K, (int)N, (int)k2, stream);
    else if (cb == 1) tf_exl3x::dequant_launch<1>(t, o, (int)K, (int)N, (int)k2, stream);
    else tf_exl3x::dequant_launch<2>(t, o, (int)K, (int)N, (int)k2, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_group_cuda(const at::Tensor& pick, at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, int64_t R,
                      int64_t slots, int64_t E) {
    TORCH_CHECK(E <= GROUP_THREADS * GROUP_PER_THREAD, "too many experts for the grouping kernel");
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    const size_t smem = (size_t)R * slots * sizeof(int);
    constexpr size_t static_smem = GROUP_THREADS / 32 * sizeof(int);
    if (smem + static_smem > 48 * 1024) {
        cudaFuncAttributes attributes;
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, group_kernel));
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const size_t limit = device->sharedMemPerBlockOptin - attributes.sharedSizeBytes;
        TORCH_CHECK(smem <= limit, "EXL3 grouping needs ", smem, " dynamic shared-memory bytes; this GPU allows ",
                    limit, " after the kernel's static storage");
        if (smem > (size_t)attributes.maxDynamicSharedSizeBytes)
            C10_CUDA_CHECK(cudaFuncSetAttribute(group_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)limit));
    }
    group_kernel<<<1, GROUP_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
        pick.data_ptr<int>(), uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), (int)R,
        (int)slots, (int)E, (int)members.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_rot_in_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& pick, const at::Tensor& suh0,
                       const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1, int64_t rows, int64_t K,
                       int64_t slots, int64_t E, bool wide4) {
    auto stream = at::cuda::getCurrentCUDAStream();
    if (wide4 && K % 512 == 0) {
        dim3 g4((unsigned)(rows * slots), (unsigned)(K / 512), 2);
        auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
        auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
        auto o0 = reinterpret_cast<half*>(out0.data_ptr());
        auto o1 = reinterpret_cast<half*>(out1.data_ptr());
        if (x.scalar_type() == at::kBFloat16)
            rot_in4_kernel<__nv_bfloat16><<<g4, 128, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x_stride, pick.data_ptr<int>(), s0, s1, o0,
                o1, (int)K, (int)slots, (int)E);
        else
            rot_in4_kernel<half><<<g4, 128, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride,
                                                         pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots,
                                                         (int)E);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    dim3 grid((unsigned)(rows * slots), (unsigned)(K / 128), 2);
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    if (x.scalar_type() == at::kBFloat16)
        rot_in_kernel<__nv_bfloat16><<<grid, 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                                              (int)x_stride, pick.data_ptr<int>(), s0, s1, o0, o1,
                                                              (int)K, (int)slots, (int)E);
    else
        rot_in_kernel<half><<<grid, 32, 0, stream>>>(reinterpret_cast<const half*>(x.data_ptr()), (int)x_stride,
                                                     pick.data_ptr<int>(), s0, s1, o0, o1, (int)K, (int)slots,
                                                     (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_gateup_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_g,
                                const at::Tensor& svh_u, const at::Tensor& suh_d, at::Tensor& xd, int64_t rows,
                                int64_t P, int64_t N, int64_t SK, int64_t slots, int64_t E, double limit,
                                int64_t act_mode) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(N / 128));
    gateup_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_g.data_ptr()),
        reinterpret_cast<const half*>(svh_u.data_ptr()), reinterpret_cast<const half*>(suh_d.data_ptr()),
        reinterpret_cast<half*>(xd.data_ptr()), (int)P, (int)N, (int)SK, (int)E, (float)limit, (int)act_mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_epilogue_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                              int64_t rows, int64_t P, int64_t D, int64_t SK, int64_t slots, int64_t E) {
    dim3 grid((unsigned)(rows * slots), (unsigned)(D / 128));
    down_epilogue_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_combine_cuda(const at::Tensor& y, const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t D,
                        int64_t slots) {
    dim3 grid((unsigned)rows, (unsigned)((D + 255) / 256));
    combine_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(y.data_ptr<float>(), wts.data_ptr<float>(),
                                                                        out.data_ptr<float>(), (int)D, (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_down_combine_cuda(const at::Tensor& Z, const at::Tensor& pick, const at::Tensor& svh_d, at::Tensor& y,
                             const at::Tensor& wts, at::Tensor& out, int64_t rows, int64_t P, int64_t D, int64_t SK,
                             int64_t slots, int64_t E) {
    TORCH_CHECK(slots <= 32, "at most 32 slots a row");
    dim3 grid((unsigned)rows, (unsigned)(D / 128));
    down_combine_kernel<<<grid, (unsigned)(32 * slots), 0, at::cuda::getCurrentCUDAStream()>>>(
        Z.data_ptr<float>(), pick.data_ptr<int>(), reinterpret_cast<const half*>(svh_d.data_ptr()),
        y.data_ptr<float>(), wts.data_ptr<float>(), out.data_ptr<float>(), (int)P, (int)D, (int)SK, (int)E,
        (int)slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exl3x_route_cuda(const at::Tensor& logits, at::Tensor& pick, at::Tensor& wts, const at::Tensor& x, int64_t x_stride,
                      const at::Tensor& suh0, const at::Tensor& suh1, at::Tensor& out0, at::Tensor& out1,
                      at::Tensor& uids, at::Tensor& ucount, at::Tensor& members, at::Tensor& counter, int64_t R,
                      int64_t K, int64_t slots, int64_t NE, int64_t topk, int64_t E) {
    TORCH_CHECK(slots == topk + 1 && slots <= 32, "route: slots = top_k + 1 <= 32");
    TORCH_CHECK(E <= ROUTE_THREADS * ROUTE_GPT, "route: too many experts");
    TORCH_CHECK(NE + 1 <= 1024 && K % 512 == 0, "route: NE + 1 <= 1024 logits and K a multiple of 512");
    int nb = 1;           // next_pow2(NE + 1), the Triton kernel's BLOCK (only NV * 32 > NE matters: pads are -inf)
    while (nb < NE + 1) nb <<= 1;
    dim3 grid((unsigned)(R * slots), (unsigned)(K / 512), 2);
    const size_t smem = (size_t)R * slots * sizeof(int);
    TORCH_CHECK(smem <= 32 * 1024, "route: too many rows");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto s0 = reinterpret_cast<const half*>(suh0.data_ptr());
    auto s1 = reinterpret_cast<const half*>(suh1.data_ptr());
    auto o0 = reinterpret_cast<half*>(out0.data_ptr());
    auto o1 = reinterpret_cast<half*>(out1.data_ptr());
    const int NL = (int)logits.size(1), maxm = (int)members.size(1);
#define TF_ROUTE_LAUNCH(TIN, NV)                                                                                    \
    route_kernel<TIN, NV><<<grid, ROUTE_THREADS, smem, stream>>>(                                                  \
        logits.data_ptr<float>(), NL, (int)NE, (int)topk, pick.data_ptr<int>(), wts.data_ptr<float>(),             \
        reinterpret_cast<const TIN*>(x.data_ptr()), (int)x_stride, s0, s1, o0, o1, (int)K, (int)slots, (int)E,     \
        uids.data_ptr<int>(), ucount.data_ptr<int>(), members.data_ptr<int>(), maxm, (int)R, counter.data_ptr<int>())
    const bool bf = x.scalar_type() == at::kBFloat16;
    switch (nb) {
        case 1024: if (bf) TF_ROUTE_LAUNCH(__nv_bfloat16, 32); else TF_ROUTE_LAUNCH(half, 32); break;
        case 512: if (bf) TF_ROUTE_LAUNCH(__nv_bfloat16, 16); else TF_ROUTE_LAUNCH(half, 16); break;
        case 256: if (bf) TF_ROUTE_LAUNCH(__nv_bfloat16, 8); else TF_ROUTE_LAUNCH(half, 8); break;
        default: if (bf) TF_ROUTE_LAUNCH(__nv_bfloat16, 4); else TF_ROUTE_LAUNCH(half, 4); break;   // nb <= 128
    }
#undef TF_ROUTE_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Flash Next's QSA indexer scores for prompt rows (W5-5), bit for bit what attention._scores computes.
//
// _scores (Triton 3.7, [64 blocks, 128 dims] tile, sizePerThread [1, 8], 16 lanes along the dims) forms each q . k as
// 16 lane partials, partial c = the products of dims 8c .. 8c + 7 added in order (the first a plain product, then
// fused adds; bf16 x bf16 products are exact in fp32), then a butterfly over the 16 lanes, xor 8, 4, 2, 1. Each head's
// sum goes through max(s, 0) and is added in head order to a total that starts at 0; the score is
// div.full.f32(total, sqrt(128)). Here one thread holds one block's 128 key values in registers and replays exactly
// those operations for RT rows (their queries in shared memory, read as broadcasts): no lane shuffles, and the key tile
// is read once for RT rows. It also leaves each 8-block group's largest select key (attention._block_keys' order) in
// smx, which bounds the cut for attention._select_cand.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float lo_bf16(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_bf16(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

__device__ __forceinline__ uint32_t select_key(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int RT>
__global__ void __launch_bounds__(128) qsa_scores_kernel(const __nv_bfloat16* __restrict__ iq,
                                                         const __nv_bfloat16* __restrict__ pooled,
                                                         const int* __restrict__ pos0p, float* __restrict__ sc,
                                                         int* __restrict__ smx, int R, int NB, int NSUB) {
    constexpr int TOP = 512, RATIO = 4, HI = 4, DI = 128;
    __shared__ __align__(16) float qs[RT * HI * DI];
    const int tid = threadIdx.x;
    const int r0 = blockIdx.x * RT;
    const int p0 = *pos0p;
    const int last = min(r0 + RT, R) - 1;
    const int hi_complete = (p0 + last + 1) / RATIO;       // the tile's last row completes the most blocks
    const int bt0 = blockIdx.y * 128;
    if (hi_complete <= TOP || bt0 >= hi_complete) return;
    for (int i = tid; i < RT * HI * DI; i += 128) {
        const int rr = i / (HI * DI);
        qs[i] = r0 + rr < R ? __bfloat162float(iq[(size_t)(r0 + rr) * HI * DI + i % (HI * DI)]) : 0.f;
    }
    const int b = bt0 + tid;
    float k[DI];
    if (b < hi_complete) {
        const uint4* src = reinterpret_cast<const uint4*>(pooled + (size_t)b * DI);
#pragma unroll
        for (int i = 0; i < DI / 8; ++i) {
            const uint4 v = __ldg(src + i);
            k[8 * i + 0] = lo_bf16(v.x); k[8 * i + 1] = hi_bf16(v.x);
            k[8 * i + 2] = lo_bf16(v.y); k[8 * i + 3] = hi_bf16(v.y);
            k[8 * i + 4] = lo_bf16(v.z); k[8 * i + 5] = hi_bf16(v.z);
            k[8 * i + 6] = lo_bf16(v.w); k[8 * i + 7] = hi_bf16(v.w);
        }
    } else {
#pragma unroll
        for (int i = 0; i < DI; ++i) k[i] = 0.f;
    }
    __syncthreads();
    const float root = __uint_as_float(0x413504F3u);      // sqrt(128.0f), as _scores' constant
    for (int rr = 0; rr < RT; ++rr) {
        const int r = r0 + rr;
        if (r >= R) break;
        const int complete = (p0 + r + 1) / RATIO;
        if (complete <= TOP) continue;                     // a dense row: _scores writes nothing
        float total = 0.f;
#pragma unroll
        for (int h = 0; h < HI; ++h) {
            const float* q = qs + (rr * HI + h) * DI;
            float P[16];
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const float4 qa = *reinterpret_cast<const float4*>(q + 8 * c);
                const float4 qb = *reinterpret_cast<const float4*>(q + 8 * c + 4);
                float a = __fmul_rn(k[8 * c + 0], qa.x);
                a = __fmaf_rn(k[8 * c + 1], qa.y, a);
                a = __fmaf_rn(k[8 * c + 2], qa.z, a);
                a = __fmaf_rn(k[8 * c + 3], qa.w, a);
                a = __fmaf_rn(k[8 * c + 4], qb.x, a);
                a = __fmaf_rn(k[8 * c + 5], qb.y, a);
                a = __fmaf_rn(k[8 * c + 6], qb.z, a);
                a = __fmaf_rn(k[8 * c + 7], qb.w, a);
                P[c] = a;
            }
#pragma unroll
            for (int c = 0; c < 8; ++c) P[c] = __fadd_rn(P[c], P[c + 8]);     // xor 8
#pragma unroll
            for (int c = 0; c < 4; ++c) P[c] = __fadd_rn(P[c], P[c + 4]);     // xor 4
#pragma unroll
            for (int c = 0; c < 2; ++c) P[c] = __fadd_rn(P[c], P[c + 2]);     // xor 2
            const float s = __fadd_rn(P[0], P[1]);                             // xor 1
            total = __fadd_rn(total, fmaxf(s, 0.f));
        }
        float v;
        asm("div.full.f32 %0, %1, %2;" : "=f"(v) : "f"(total), "f"(root));
        const bool ok = b < complete;
        if (ok) sc[(size_t)r * NB + b] = v;
        if (smx != nullptr) {
            uint32_t key = ok ? select_key(v) : 0u;
            key = max(key, __shfl_xor_sync(0xFFFFFFFFu, key, 1));
            key = max(key, __shfl_xor_sync(0xFFFFFFFFu, key, 2));
            key = max(key, __shfl_xor_sync(0xFFFFFFFFu, key, 4));
            if ((tid & 7) == 0 && b < complete) smx[(size_t)r * NSUB + b / 8] = (int)key;
        }
    }
}

}  // namespace

void qsa_scores_cuda(const __nv_bfloat16* iq, const __nv_bfloat16* pooled, const int* pos0, float* sc, int* smx, int R,
                     int NB, int NSUB, int blocks, int rt, cudaStream_t stream) {
    const dim3 block(128);
    if (rt == 4) {
        const dim3 grid((R + 3) / 4, (blocks + 127) / 128);
        qsa_scores_kernel<4><<<grid, block, 0, stream>>>(iq, pooled, pos0, sc, smx, R, NB, NSUB);
    } else if (rt == 16) {
        const dim3 grid((R + 15) / 16, (blocks + 127) / 128);
        qsa_scores_kernel<16><<<grid, block, 0, stream>>>(iq, pooled, pos0, sc, smx, R, NB, NSUB);
    } else {
        const dim3 grid((R + 7) / 8, (blocks + 127) / 128);
        qsa_scores_kernel<8><<<grid, block, 0, stream>>>(iq, pooled, pos0, sc, smx, R, NB, NSUB);
    }
}

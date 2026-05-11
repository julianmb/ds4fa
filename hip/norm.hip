#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"

#define WAVEFRONT_SIZE 32

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    return val;
}

typedef float4 float4_t;

__device__ inline float dot_f4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

extern "C" __global__ void kernel_rms_norm_fuse_impl(
    struct ds4_hip_args_norm args,
    const char * src0,
    const char * src1_0,
    const char * src1_1,
    char * dst,
    int F) 
{
    extern __shared__ float shmem_f32[];

    const int i01 = blockIdx.x;
    const int i02 = blockIdx.y;
    const int i03 = blockIdx.z;

    const float4_t * x = (const float4_t *) (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    const float4_t * f0 = (const float4_t *) (src1_0 + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    const float4_t * f1 = (const float4_t *) (src1_1 + (i03%args.nef3[2])*args.nbf3[2] + (i02%args.nef2[2])*args.nbf2[2] + (i01%args.nef1[2])*args.nbf1[2]);

    float sumf = 0.0f;

    for (int i00 = threadIdx.x; i00 < args.ne00_t; i00 += blockDim.x) {
        float4_t val = x[i00];
        sumf += dot_f4(val, val);
    }
    
    sumf = warp_reduce_sum(sumf);

    const int lane = threadIdx.x % WAVEFRONT_SIZE;
    const int wid = threadIdx.x / WAVEFRONT_SIZE;

    if (lane == 0) {
        shmem_f32[wid] = sumf;
    }

    __syncthreads();

    sumf = (threadIdx.x < blockDim.x / WAVEFRONT_SIZE) ? shmem_f32[lane] : 0.0f;
    if (wid == 0) {
        sumf = warp_reduce_sum(sumf);
        shmem_f32[0] = sumf;
    }

    __syncthreads();

    sumf = shmem_f32[0];
    const float mean  = sumf / args.ne00;
    const float scale = 1.0f / sqrtf(mean + args.eps);

    float4_t * y = (float4_t *) (dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    for (int i00 = threadIdx.x; i00 < args.ne00_t; i00 += blockDim.x) {
        float4_t val = x[i00];
        if (F == 1) {
            val.x *= scale; val.y *= scale; val.z *= scale; val.w *= scale;
            y[i00] = val;
        } else if (F == 2) {
            float4_t w0 = f0[i00];
            val.x *= scale * w0.x; val.y *= scale * w0.y; val.z *= scale * w0.z; val.w *= scale * w0.w;
            y[i00] = val;
        } else if (F == 3) {
            float4_t w0 = f0[i00];
            float4_t w1 = f1[i00];
            val.x = (val.x * scale) * w0.x + w1.x;
            val.y = (val.y * scale) * w0.y + w1.y;
            val.z = (val.z * scale) * w0.z + w1.z;
            val.w = (val.w * scale) * w0.w + w1.w;
            y[i00] = val;
        }
    }
}

extern "C" __global__ void kernel_dsv4_qkv_rms_norm_f32_4(
    struct ds4_hip_args_qkv_rms_norm args,
    const float4 * q_src,
    const float4 * q_weight,
    float4 * q_dst,
    const float4 * kv_src,
    const float4 * kv_weight,
    float4 * kv_dst)
{
    extern __shared__ float shmem_f32[];

    const uint row = blockIdx.x;
    const bool kv_task = blockIdx.y != 0;
    const int n = kv_task ? args.kv_n : args.q_n;
    const int n4 = kv_task ? args.kv_n4 : args.q_n4;
    const uint64_t row_stride4 = (kv_task ? args.kv_row_stride : args.q_row_stride) / sizeof(float4);

    const float4 * x = kv_task ? kv_src + row * row_stride4 : q_src + row * row_stride4;
    const float4 * w = kv_task ? kv_weight : q_weight;
    float4 * y = kv_task ? kv_dst + row * row_stride4 : q_dst + row * row_stride4;

    float sumf = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x[i];
        sumf += dot_f4(v, v);
    }
    
    sumf = warp_reduce_sum(sumf);

    const int lane = threadIdx.x % WAVEFRONT_SIZE;
    const int wid = threadIdx.x / WAVEFRONT_SIZE;

    if (lane == 0) {
        shmem_f32[wid] = sumf;
    }

    __syncthreads();

    sumf = (threadIdx.x < blockDim.x / WAVEFRONT_SIZE) ? shmem_f32[lane] : 0.0f;
    if (wid == 0) {
        sumf = warp_reduce_sum(sumf);
        shmem_f32[0] = sumf;
    }

    __syncthreads();

    sumf = shmem_f32[0];
    const float scale = rsqrtf(sumf / (float)n + args.eps);

    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 v = x[i];
        float4 weight = w[i];
        v.x *= scale * weight.x;
        v.y *= scale * weight.y;
        v.z *= scale * weight.z;
        v.w *= scale * weight.w;
        y[i] = v;
    }
}

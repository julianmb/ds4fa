#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 32
#endif

__device__ inline float warp_max(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset, WAVEFRONT_SIZE));
    }
    return val;
}

__device__ inline float warp_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

template<typename T>
__device__ void kernel_soft_max_impl(
        ds4_hip_args_soft_max args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    const int32_t i03 = blockIdx.z;
    const int32_t i02 = blockIdx.y;
    const int32_t i01 = blockIdx.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    const float * psrc0 =                (const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const     T * pmask = src1 != src0 ? (const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    const float * psrc2 = src2 != src0 ? (const float *) (src2)                                                 : nullptr;
          float * pdst  =                (      float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    extern __shared__ float buf[];
    int sgitg = threadIdx.x / WAVEFRONT_SIZE;
    int tiisg = threadIdx.x % WAVEFRONT_SIZE;

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = powf(base, exp);
    }

    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = threadIdx.x; i00 < args.ne00; i00 += blockDim.x) {
        lmax = fmaxf(lmax, psrc0[i00]*args.scale + (pmask ? slope*(float)pmask[i00] : 0.0f));
    }

    float max_val = warp_max(lmax);
    if (blockDim.x > WAVEFRONT_SIZE) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        __syncthreads();

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        __syncthreads();

        max_val = buf[tiisg];
        max_val = warp_max(max_val);
    }

    float lsum = 0.0f;
    for (int i00 = threadIdx.x; i00 < args.ne00; i00 += blockDim.x) {
        const float exp_psrc0 = expf((psrc0[i00]*args.scale + (pmask ? slope*(float)pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    __syncthreads();

    float sum = warp_sum(lsum);

    if (blockDim.x > WAVEFRONT_SIZE) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        __syncthreads();

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        __syncthreads();

        sum = buf[tiisg];
        sum = warp_sum(sum);
    }

    if (psrc2) {
        sum += expf(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = threadIdx.x; i00 < args.ne00; i00 += blockDim.x) {
        pdst[i00] *= inv_sum;
    }
}

extern "C" __global__ void kernel_soft_max_f32(
        ds4_hip_args_soft_max args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    kernel_soft_max_impl<float>(args, src0, src1, src2, dst);
}

template<typename T>
__device__ void kernel_soft_max_4_impl(
        ds4_hip_args_soft_max args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    const int32_t i03 = blockIdx.z;
    const int32_t i02 = blockIdx.y;
    const int32_t i01 = blockIdx.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    const float4 * psrc4 =                (const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const      T * pmask = src1 != src0 ? (const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    const float  * psrc2 = src2 != src0 ? (const float * ) (src2)                                                 : nullptr;
          float4 * pdst4 =                (      float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    extern __shared__ float buf[];
    int sgitg = threadIdx.x / WAVEFRONT_SIZE;
    int tiisg = threadIdx.x % WAVEFRONT_SIZE;

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp_val  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = powf(base, exp_val);
    }

    float4 lmax4;
    lmax4.x = psrc2 ? psrc2[i02] : -INFINITY;
    lmax4.y = lmax4.x;
    lmax4.z = lmax4.x;
    lmax4.w = lmax4.x;

    for (int i00 = threadIdx.x; i00 < args.ne00/4; i00 += blockDim.x) {
        float4 src_val = psrc4[i00];
        float4 m4;
        if (pmask) {
            m4 = pmask[i00];
            m4.x *= slope; m4.y *= slope; m4.z *= slope; m4.w *= slope;
        } else {
            m4.x = 0.0f; m4.y = 0.0f; m4.z = 0.0f; m4.w = 0.0f;
        }
        lmax4.x = fmaxf(lmax4.x, src_val.x*args.scale + m4.x);
        lmax4.y = fmaxf(lmax4.y, src_val.y*args.scale + m4.y);
        lmax4.z = fmaxf(lmax4.z, src_val.z*args.scale + m4.z);
        lmax4.w = fmaxf(lmax4.w, src_val.w*args.scale + m4.w);
    }

    float lmax = fmaxf(fmaxf(lmax4.x, lmax4.y), fmaxf(lmax4.z, lmax4.w));

    float max_val = warp_max(lmax);
    if (blockDim.x > WAVEFRONT_SIZE) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        __syncthreads();

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        __syncthreads();

        max_val = buf[tiisg];
        max_val = warp_max(max_val);
    }

    float4 lsum4;
    lsum4.x = 0.0f; lsum4.y = 0.0f; lsum4.z = 0.0f; lsum4.w = 0.0f;
    for (int i00 = threadIdx.x; i00 < args.ne00/4; i00 += blockDim.x) {
        float4 src_val = psrc4[i00];
        float4 m4;
        if (pmask) {
            m4 = pmask[i00];
            m4.x *= slope; m4.y *= slope; m4.z *= slope; m4.w *= slope;
        } else {
            m4.x = 0.0f; m4.y = 0.0f; m4.z = 0.0f; m4.w = 0.0f;
        }
        float4 exp_psrc4;
        exp_psrc4.x = expf((src_val.x*args.scale + m4.x) - max_val);
        exp_psrc4.y = expf((src_val.y*args.scale + m4.y) - max_val);
        exp_psrc4.z = expf((src_val.z*args.scale + m4.z) - max_val);
        exp_psrc4.w = expf((src_val.w*args.scale + m4.w) - max_val);

        lsum4.x += exp_psrc4.x;
        lsum4.y += exp_psrc4.y;
        lsum4.z += exp_psrc4.z;
        lsum4.w += exp_psrc4.w;
        pdst4[i00] = exp_psrc4;
    }

    float lsum = lsum4.x + lsum4.y + lsum4.z + lsum4.w;

    __syncthreads();

    float sum = warp_sum(lsum);

    if (blockDim.x > WAVEFRONT_SIZE) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        __syncthreads();

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        __syncthreads();

        sum = buf[tiisg];
        sum = warp_sum(sum);
    }

    if (psrc2) {
        sum += expf(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = threadIdx.x; i00 < args.ne00/4; i00 += blockDim.x) {
        float4 dst_val = pdst4[i00];
        dst_val.x *= inv_sum;
        dst_val.y *= inv_sum;
        dst_val.z *= inv_sum;
        dst_val.w *= inv_sum;
        pdst4[i00] = dst_val;
    }
}

extern "C" __global__ void kernel_soft_max_f32_4(
        ds4_hip_args_soft_max args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst) {
    kernel_soft_max_4_impl<float4>(args, src0, src1, src2, dst);
}

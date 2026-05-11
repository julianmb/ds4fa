#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 32
#endif

__device__ inline float warp_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    }
    return val;
}

template <typename T0, typename T>
__device__ void kernel_sum_rows_impl(
        ds4_hip_args_sum_rows args,
        const char * src0,
        char * dst) {
    const int i3 = blockIdx.z;
    const int i2 = blockIdx.y;
    const int i1 = blockIdx.x;

    extern __shared__ char shmem[];
    T0 * shmem_t = (T0 *) shmem;

    int sgitg = threadIdx.x / WAVEFRONT_SIZE;
    int tiisg = threadIdx.x % WAVEFRONT_SIZE;

    if (sgitg == 0) {
        shmem_t[tiisg] = 0.0f;
    }

    const T0 * src_row = (const T0 *) (src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
          T  * dst_row = (      T  *) (dst  + i1*args.nb1  + i2*args.nb2  + i3*args.nb3);

    T0 sumf = (T0)(0.0f);

    for (int64_t i0 = threadIdx.x; i0 < args.ne00; i0 += blockDim.x) {
        sumf += src_row[i0];
    }

    sumf = warp_sum(sumf);

    __syncthreads();

    if (tiisg == 0) {
        shmem_t[sgitg] = sumf;
    }

    __syncthreads();

    sumf = shmem_t[tiisg];
    sumf = warp_sum(sumf);

    if (threadIdx.x == 0) {
        if (args.op == OP_SUM_ROWS_NUM_MEAN) {
            dst_row[0] = sumf / args.ne00;
        } else {
            dst_row[0] = sumf;
        }
    }
}

extern "C" __global__ void kernel_sum_rows_f32_f32(
        ds4_hip_args_sum_rows args,
        const char * src0,
        char * dst) {
    kernel_sum_rows_impl<float, float>(args, src0, dst);
}

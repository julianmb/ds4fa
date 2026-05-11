#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template<typename T>
__device__ void kernel_repeat_impl(
        ds4_hip_args_repeat args,
        const char * src0,
        char * dst) {
    const int i3 = blockIdx.z;
    const int i2 = blockIdx.y;
    const int i1 = blockIdx.x;

    const int i03 = i3%args.ne03;
    const int i02 = i2%args.ne02;
    const int i01 = i1%args.ne01;

    const char * src0_ptr = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;
          char * dst_ptr  = dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1;

    for (int i0 = threadIdx.x; i0 < args.ne0; i0 += blockDim.x) {
        const int i00 = i0%args.ne00;
        *((T *)(dst_ptr + i0*args.nb0)) = *((const T *)(src0_ptr + i00*args.nb00));
    }
}

extern "C" __global__ void kernel_repeat_f32(
        ds4_hip_args_repeat args,
        const char * src0,
        char * dst) {
    kernel_repeat_impl<float>(args, src0, dst);
}

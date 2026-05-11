#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"

template<typename T0, typename T1>
__device__ void kernel_cpy_t_t_impl(
        struct ds4_hip_args_cpy args,
        const char * src0,
        char * dst) {
    const int i03 = blockIdx.z;
    const int i02 = blockIdx.y;
    const int i01 = blockDim.y == 1 ? blockIdx.x % args.ne01 : blockIdx.x * blockDim.y + threadIdx.y;
    const int iw0 = blockDim.y == 1 ? blockIdx.x / args.ne01 : 0;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int64_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int64_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int64_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int64_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    T1 * dst_data = (T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int64_t i00 = iw0*blockDim.x + threadIdx.x; i00 < args.ne00; ) {
        const T0 * src = (const T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break; // Following original logic
    }
}

extern "C" __global__ void kernel_cpy_f32_f32(
        struct ds4_hip_args_cpy args, const char * src0, char * dst) {
    kernel_cpy_t_t_impl<float, float>(args, src0, dst);
}

extern "C" __global__ void kernel_cpy_f32_f16(
        struct ds4_hip_args_cpy args, const char * src0, char * dst) {
    kernel_cpy_t_t_impl<float, __half>(args, src0, dst);
}

extern "C" __global__ void kernel_cpy_f16_f32(
        struct ds4_hip_args_cpy args, const char * src0, char * dst) {
    kernel_cpy_t_t_impl<__half, float>(args, src0, dst);
}

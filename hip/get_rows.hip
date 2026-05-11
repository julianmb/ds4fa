#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template<typename T0, typename T>
__device__ void kernel_get_rows_impl(
        ds4_hip_args_get_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    const int32_t iw0 = blockIdx.x / args.ne10;
    const int32_t i10 = blockIdx.x % args.ne10;
    const int32_t i11 = blockIdx.y;
    const int32_t i12 = blockIdx.z;

    const int32_t r = ((const int32_t *) ((const char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    const T0 * psrc = (const T0 *) ((const char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
          T  * pdst = (      T  *) ((      char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*blockDim.x + threadIdx.x; ind < args.ne00t;) {
        pdst[ind] = (T) psrc[ind];
        break;
    }
}

extern "C" __global__ void kernel_get_rows_f32(
        ds4_hip_args_get_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    kernel_get_rows_impl<float, float>(args, src0, src1, dst);
}

extern "C" __global__ void kernel_get_rows_f16(
        ds4_hip_args_get_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    kernel_get_rows_impl<half, float>(args, src0, src1, dst);
}

extern "C" __global__ void kernel_get_rows_i32(
        ds4_hip_args_get_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    kernel_get_rows_impl<int32_t, int32_t>(args, src0, src1, dst);
}

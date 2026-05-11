#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

template<typename T, typename TI>
__device__ void kernel_set_rows_impl(
        ds4_hip_args_set_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    const int32_t i03 = blockIdx.z;
    const int32_t i02 = blockIdx.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = blockIdx.x*blockDim.y + threadIdx.y;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const TI *) ((const char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          T     * dst_row = (      T     *) ((      char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const float * src_row = (const float *) ((const char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = threadIdx.x; ind < args.nk0; ind += blockDim.x) {
        dst_row[ind] = (T) src_row[ind];
    }
}

extern "C" __global__ void kernel_set_rows_f32_i32(
        ds4_hip_args_set_rows args,
        const void * src0,
        const void * src1,
        void * dst) {
    kernel_set_rows_impl<float, int32_t>(args, src0, src1, dst);
}

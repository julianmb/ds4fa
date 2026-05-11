#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void kernel_swiglu_f32(
        ds4_hip_args_glu args,
        const char * src0,
        const char * src1,
        char * dst) {
    const float * src0_row = (const float *) (src0 + blockIdx.x*args.nb01) + args.i00;
    const float * src1_row = (const float *) (src1 + blockIdx.x*args.nb11) + args.i10;
          float * dst_row  = (      float *) (dst  + blockIdx.x*args.nb1);

    for (int i0 = threadIdx.x; i0 < args.ne0; i0 += blockDim.x) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + expf(-x0));

        dst_row[i0] = silu*x1;
    }
}

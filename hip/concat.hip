#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"

extern "C" __global__ void kernel_concat(
        struct ds4_hip_args_concat args,
        const char * src0,
        const char * src1,
        char * dst) {
    const int i3 = blockIdx.z;
    const int i2 = blockIdx.y;
    const int i1 = blockIdx.x;

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    const float * x;

    for (int i0 = threadIdx.x; i0 < args.ne0; i0 += blockDim.x) {
        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (const float *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (const float *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        float * y = (float *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

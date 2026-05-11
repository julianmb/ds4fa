#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"

template <typename T0, typename T1, typename T, short FC_OP, short FC_F, bool FC_RB, bool FC_CB>
__device__ void kernel_bin_fuse_impl(
        struct ds4_hip_args_bin args,
        const char * src0,
        const char * src1,
        char * dst) {

    if (FC_RB) {
        const uint i0 = blockIdx.y * args.ne00 + blockIdx.x;
        const uint i1 = FC_CB ? blockIdx.x % args.ne10 : blockIdx.x;

        const T0 * src0_row = (const T0 *) (src0);
        T  * dst_row  = (T  *) (dst);

        if (FC_F == 1) {
            const T1 * src1_row = (const T1 *) (src1 + args.o1[0]);

            if (FC_OP == 0) {
                dst_row[i0] = src0_row[i0] + src1_row[i1];
            }

            if (FC_OP == 1) {
                dst_row[i0] = src0_row[i0] - src1_row[i1];
            }

            if (FC_OP == 2) {
                dst_row[i0] = src0_row[i0] * src1_row[i1];
            }

            if (FC_OP == 3) {
                dst_row[i0] = src0_row[i0] / src1_row[i1];
            }
        } else {
            T0 res = src0_row[i0];

            if (FC_OP == 0) {
                #pragma unroll
                for (short j = 0; j < FC_F; ++j) {
                    res += ((const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 1) {
                #pragma unroll
                for (short j = 0; j < FC_F; ++j) {
                    res -= ((const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 2) {
                #pragma unroll
                for (short j = 0; j < FC_F; ++j) {
                    res *= ((const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            if (FC_OP == 3) {
                #pragma unroll
                for (short j = 0; j < FC_F; ++j) {
                    res /= ((const T1 *) (src1 + args.o1[j]))[i1];
                }
            }

            dst_row[i0] = res;
        }
    } else {
        const int i03 = blockIdx.z;
        const int i02 = blockIdx.y;
        const int i01 = blockIdx.x;

        if (i01 >= args.ne01) {
            return;
        }

        const int i13 = i03 % args.ne13;
        const int i12 = i02 % args.ne12;
        const int i11 = i01 % args.ne11;

        const T0 * src0_ptr = (const T0 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + args.offs);
        T  * dst_ptr  = (T  *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1  + args.offs);

        if (FC_F == 1) {
            const T1 * src1_ptr = (const T1 *) (src1 + args.o1[0] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);

            for (int i0 = threadIdx.x; i0 < args.ne0; i0 += blockDim.x) {
                const int i10 = FC_CB ? i0 % args.ne10 : i0;

                if (FC_OP == 0) {
                    dst_ptr[i0] = src0_ptr[i0] + src1_ptr[i10];
                }

                if (FC_OP == 1) {
                    dst_ptr[i0] = src0_ptr[i0] - src1_ptr[i10];
                }

                if (FC_OP == 2) {
                    dst_ptr[i0] = src0_ptr[i0] * src1_ptr[i10];
                }

                if (FC_OP == 3) {
                    dst_ptr[i0] = src0_ptr[i0] / src1_ptr[i10];
                }
            }
        } else {
            const T1 * src1_ptr[8];
            #pragma unroll
            for (short j = 0; j < FC_F; ++j) {
                src1_ptr[j] = (const T1 *) (src1 + args.o1[j] + i13*args.nb13 + i12*args.nb12 + i11*args.nb11);
            }

            for (int i0 = threadIdx.x; i0 < args.ne0; i0 += blockDim.x) {
                const int i10 = FC_CB ? i0 % args.ne10 : i0;

                T res = src0_ptr[i0];

                if (FC_OP == 0) {
                    #pragma unroll
                    for (short j = 0; j < FC_F; ++j) {
                        res += src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 1) {
                    #pragma unroll
                    for (short j = 0; j < FC_F; ++j) {
                        res -= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 2) {
                    #pragma unroll
                    for (short j = 0; j < FC_F; ++j) {
                        res *= src1_ptr[j][i10];
                    }
                }

                if (FC_OP == 3) {
                    #pragma unroll
                    for (short j = 0; j < FC_F; ++j) {
                        res /= src1_ptr[j][i10];
                    }
                }

                dst_ptr[i0] = res;
            }
        }
    }
}

// op=0, f=1, rb=false, cb=false
extern "C" __global__ void kernel_bin_fuse_f32_f32_f32_add(
        struct ds4_hip_args_bin args, const char * src0, const char * src1, char * dst) {
    kernel_bin_fuse_impl<float, float, float, 0, 1, false, false>(args, src0, src1, dst);
}

// op=2, f=1, rb=false, cb=false
extern "C" __global__ void kernel_bin_fuse_f32_f32_f32_mul(
        struct ds4_hip_args_bin args, const char * src0, const char * src1, char * dst) {
    kernel_bin_fuse_impl<float, float, float, 2, 1, false, false>(args, src0, src1, dst);
}

// op=2, f=1, rb=false, cb=true
extern "C" __global__ void kernel_bin_fuse_f32_f32_f32_mul_scalar(
        struct ds4_hip_args_bin args, const char * src0, const char * src1, char * dst) {
    kernel_bin_fuse_impl<float, float, float, 2, 1, false, true>(args, src0, src1, dst);
}

// op=3, f=1, rb=false, cb=true
extern "C" __global__ void kernel_bin_fuse_f32_f32_f32_div_row(
        struct ds4_hip_args_bin args, const char * src0, const char * src1, char * dst) {
    kernel_bin_fuse_impl<float, float, float, 3, 1, false, true>(args, src0, src1, dst);
}

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"

#define SWAP(x, y) do { auto tmp = (x); (x) = (y); (y) = tmp; } while(0)
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

template<ds4_sort_order order>
__device__ void kernel_argsort_f32_i32_impl(
        struct ds4_hip_args_argsort args,
        const char * src0,
        int32_t * dst) {
    
    extern __shared__ int32_t shmem_i32[];

    const int col = threadIdx.x;
    const int ib  = blockIdx.x / args.ne01;

    const int i00 = ib * blockDim.x;
    const int i01 = blockIdx.x % args.ne01;
    const int i02 = blockIdx.y;
    const int i03 = blockIdx.z;

    const float * src0_row = (const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    shmem_i32[col] = i00 + col;

    __syncthreads();

    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == DS4_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            __syncthreads();
        }
    }

    const int64_t i0 = ib * args.top_k;

    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;
        dst[col] = shmem_i32[col];
    }
}

extern "C" __global__ void kernel_argsort_f32_i32_desc(
        struct ds4_hip_args_argsort args,
        const char * src0,
        int32_t * dst) {
    kernel_argsort_f32_i32_impl<DS4_SORT_ORDER_DESC>(args, src0, dst);
}

template<ds4_sort_order order>
__device__ void kernel_argsort_merge_f32_i32_impl(
        struct ds4_hip_args_argsort_merge args,
        const char    * src0,
        const int32_t * tmp,
        int32_t * dst) {

    const int im  = blockIdx.x / args.ne01;
    const int i01 = blockIdx.x % args.ne01;
    const int i02 = blockIdx.y;
    const int i03 = blockIdx.z;

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    const float * src0_row = (const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + blockDim.x - 1) / blockDim.x;

    const int k0 = threadIdx.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

extern "C" __global__ void kernel_argsort_merge_f32_i32_desc(
        struct ds4_hip_args_argsort_merge args,
        const char    * src0,
        const int32_t * tmp,
        int32_t * dst) {
    kernel_argsort_merge_f32_i32_impl<DS4_SORT_ORDER_DESC>(args, src0, tmp, dst);
}

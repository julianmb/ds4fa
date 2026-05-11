#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"
#include <math.h>

#define WAVEFRONT_SIZE 32
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define FC_mul_mv_nsg 4 // Assuming 4 sub-groups per block as in Metal? Let's verify, but usually it's blockDim.x / 32


__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    return val;
}

typedef float4 float4_t;

__device__ inline float dot_f4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

extern "C" __global__ void kernel_dsv4_hc_split_sinkhorn(
        const struct ds4_hip_args_dsv4_hc_split_sinkhorn args,
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    const float * mix = mixes + tid * args.mix_hc;
    float * out = dst + tid * args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z_base = *((const float4 *) base);
        const float4 pre_z_mix  = *((const float4 *) mix);
        float4 pre_z;
        pre_z.x = pre_z_mix.x * pre_scale + pre_z_base.x;
        pre_z.y = pre_z_mix.y * pre_scale + pre_z_base.y;
        pre_z.z = pre_z_mix.z * pre_scale + pre_z_base.z;
        pre_z.w = pre_z_mix.w * pre_scale + pre_z_base.w;
        
        float4 pre_out;
        pre_out.x = 1.0f / (1.0f + expf(-pre_z.x)) + epsv;
        pre_out.y = 1.0f / (1.0f + expf(-pre_z.y)) + epsv;
        pre_out.z = 1.0f / (1.0f + expf(-pre_z.z)) + epsv;
        pre_out.w = 1.0f / (1.0f + expf(-pre_z.w)) + epsv;
        *((float4 *) out) = pre_out;

        const float4 post_z_base = *((const float4 *) (base + 4));
        const float4 post_z_mix  = *((const float4 *) (mix  + 4));
        float4 post_z;
        post_z.x = post_z_mix.x * post_scale + post_z_base.x;
        post_z.y = post_z_mix.y * post_scale + post_z_base.y;
        post_z.z = post_z_mix.z * post_scale + post_z_base.z;
        post_z.w = post_z_mix.w * post_scale + post_z_base.w;

        float4 post_out;
        post_out.x = 2.0f / (1.0f + expf(-post_z.x));
        post_out.y = 2.0f / (1.0f + expf(-post_z.y));
        post_out.z = 2.0f / (1.0f + expf(-post_z.z));
        post_out.w = 2.0f / (1.0f + expf(-post_z.w));
        *((float4 *) (out + 4)) = post_out;

        const float4 r0_base = *((const float4 *) (base + 8));
        const float4 r0_mix  = *((const float4 *) (mix + 8));
        float4 r0;
        r0.x = r0_mix.x * comb_scale + r0_base.x;
        r0.y = r0_mix.y * comb_scale + r0_base.y;
        r0.z = r0_mix.z * comb_scale + r0_base.z;
        r0.w = r0_mix.w * comb_scale + r0_base.w;

        const float4 r1_base = *((const float4 *) (base + 12));
        const float4 r1_mix  = *((const float4 *) (mix + 12));
        float4 r1;
        r1.x = r1_mix.x * comb_scale + r1_base.x;
        r1.y = r1_mix.y * comb_scale + r1_base.y;
        r1.z = r1_mix.z * comb_scale + r1_base.z;
        r1.w = r1_mix.w * comb_scale + r1_base.w;

        const float4 r2_base = *((const float4 *) (base + 16));
        const float4 r2_mix  = *((const float4 *) (mix + 16));
        float4 r2;
        r2.x = r2_mix.x * comb_scale + r2_base.x;
        r2.y = r2_mix.y * comb_scale + r2_base.y;
        r2.z = r2_mix.z * comb_scale + r2_base.z;
        r2.w = r2_mix.w * comb_scale + r2_base.w;

        const float4 r3_base = *((const float4 *) (base + 20));
        const float4 r3_mix  = *((const float4 *) (mix + 20));
        float4 r3;
        r3.x = r3_mix.x * comb_scale + r3_base.x;
        r3.y = r3_mix.y * comb_scale + r3_base.y;
        r3.z = r3_mix.z * comb_scale + r3_base.z;
        r3.w = r3_mix.w * comb_scale + r3_base.w;

        const float m0 = fmaxf(fmaxf(r0.x, r0.y), fmaxf(r0.z, r0.w));
        const float m1 = fmaxf(fmaxf(r1.x, r1.y), fmaxf(r1.z, r1.w));
        const float m2 = fmaxf(fmaxf(r2.x, r2.y), fmaxf(r2.z, r2.w));
        const float m3 = fmaxf(fmaxf(r3.x, r3.y), fmaxf(r3.z, r3.w));

        r0.x = expf(r0.x - m0); r0.y = expf(r0.y - m0); r0.z = expf(r0.z - m0); r0.w = expf(r0.w - m0);
        r1.x = expf(r1.x - m1); r1.y = expf(r1.y - m1); r1.z = expf(r1.z - m1); r1.w = expf(r1.w - m1);
        r2.x = expf(r2.x - m2); r2.y = expf(r2.y - m2); r2.z = expf(r2.z - m2); r2.w = expf(r2.w - m2);
        r3.x = expf(r3.x - m3); r3.y = expf(r3.y - m3); r3.z = expf(r3.z - m3); r3.w = expf(r3.w - m3);

        const float r0_sum = 1.0f / (r0.x + r0.y + r0.z + r0.w);
        r0.x = r0.x * r0_sum + epsv; r0.y = r0.y * r0_sum + epsv; r0.z = r0.z * r0_sum + epsv; r0.w = r0.w * r0_sum + epsv;
        
        const float r1_sum = 1.0f / (r1.x + r1.y + r1.z + r1.w);
        r1.x = r1.x * r1_sum + epsv; r1.y = r1.y * r1_sum + epsv; r1.z = r1.z * r1_sum + epsv; r1.w = r1.w * r1_sum + epsv;
        
        const float r2_sum = 1.0f / (r2.x + r2.y + r2.z + r2.w);
        r2.x = r2.x * r2_sum + epsv; r2.y = r2.y * r2_sum + epsv; r2.z = r2.z * r2_sum + epsv; r2.w = r2.w * r2_sum + epsv;
        
        const float r3_sum = 1.0f / (r3.x + r3.y + r3.z + r3.w);
        r3.x = r3.x * r3_sum + epsv; r3.y = r3.y * r3_sum + epsv; r3.z = r3.z * r3_sum + epsv; r3.w = r3.w * r3_sum + epsv;

        float4 col_inv;
        col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
        col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
        col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
        col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

        r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
        r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
        r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
        r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            const float r0_sum_it = 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r0.x *= r0_sum_it; r0.y *= r0_sum_it; r0.z *= r0_sum_it; r0.w *= r0_sum_it;
            
            const float r1_sum_it = 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r1.x *= r1_sum_it; r1.y *= r1_sum_it; r1.z *= r1_sum_it; r1.w *= r1_sum_it;
            
            const float r2_sum_it = 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r2.x *= r2_sum_it; r2.y *= r2_sum_it; r2.z *= r2_sum_it; r2.w *= r2_sum_it;
            
            const float r3_sum_it = 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            r3.x *= r3_sum_it; r3.y *= r3_sum_it; r3.z *= r3_sum_it; r3.w *= r3_sum_it;

            col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
            col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
            col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
            col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

            r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
            r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
            r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
            r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;
        }

        *((float4 *) (out +  8)) = r0;
        *((float4 *) (out + 12)) = r1;
        *((float4 *) (out + 16)) = r2;
        *((float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

extern "C" __global__ void kernel_dsv4_hc_split_weighted_sum(
        const struct ds4_hip_args_dsv4_hc_split_weighted_sum args,
        const char  * mixes,
        const float * scale,
        const float * base,
        const char  * x,
        char  * split,
        char  * dst) {
    extern __shared__ float pre_shmem[];
    uint row = blockIdx.x;
    uint tid = threadIdx.x;
    uint ntg = blockDim.x;

    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    const float * mix = (const float *) (mixes + (uint64_t)row*args.nb_mix1);
    float * out = (float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z_mix  = *((const float4 *) mix);
        const float4 pre_z_base = *((const float4 *) base);
        float4 pre_z;
        pre_z.x = pre_z_mix.x * pre_scale + pre_z_base.x;
        pre_z.y = pre_z_mix.y * pre_scale + pre_z_base.y;
        pre_z.z = pre_z_mix.z * pre_scale + pre_z_base.z;
        pre_z.w = pre_z_mix.w * pre_scale + pre_z_base.w;

        float4 pre;
        pre.x = 1.0f / (1.0f + expf(-pre_z.x)) + epsv;
        pre.y = 1.0f / (1.0f + expf(-pre_z.y)) + epsv;
        pre.z = 1.0f / (1.0f + expf(-pre_z.z)) + epsv;
        pre.w = 1.0f / (1.0f + expf(-pre_z.w)) + epsv;
        *((float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z_mix  = *((const float4 *) (mix  + 4));
        const float4 post_z_base = *((const float4 *) (base + 4));
        float4 post_z;
        post_z.x = post_z_mix.x * post_scale + post_z_base.x;
        post_z.y = post_z_mix.y * post_scale + post_z_base.y;
        post_z.z = post_z_mix.z * post_scale + post_z_base.z;
        post_z.w = post_z_mix.w * post_scale + post_z_base.w;

        float4 post_out;
        post_out.x = 2.0f / (1.0f + expf(-post_z.x));
        post_out.y = 2.0f / (1.0f + expf(-post_z.y));
        post_out.z = 2.0f / (1.0f + expf(-post_z.z));
        post_out.w = 2.0f / (1.0f + expf(-post_z.w));
        *((float4 *) (out + 4)) = post_out;

        const float4 r0_mix  = *((const float4 *) (mix  +  8));
        const float4 r0_base = *((const float4 *) (base +  8));
        float4 r0;
        r0.x = r0_mix.x * comb_scale + r0_base.x;
        r0.y = r0_mix.y * comb_scale + r0_base.y;
        r0.z = r0_mix.z * comb_scale + r0_base.z;
        r0.w = r0_mix.w * comb_scale + r0_base.w;

        const float4 r1_mix  = *((const float4 *) (mix  + 12));
        const float4 r1_base = *((const float4 *) (base + 12));
        float4 r1;
        r1.x = r1_mix.x * comb_scale + r1_base.x;
        r1.y = r1_mix.y * comb_scale + r1_base.y;
        r1.z = r1_mix.z * comb_scale + r1_base.z;
        r1.w = r1_mix.w * comb_scale + r1_base.w;

        const float4 r2_mix  = *((const float4 *) (mix  + 16));
        const float4 r2_base = *((const float4 *) (base + 16));
        float4 r2;
        r2.x = r2_mix.x * comb_scale + r2_base.x;
        r2.y = r2_mix.y * comb_scale + r2_base.y;
        r2.z = r2_mix.z * comb_scale + r2_base.z;
        r2.w = r2_mix.w * comb_scale + r2_base.w;

        const float4 r3_mix  = *((const float4 *) (mix  + 20));
        const float4 r3_base = *((const float4 *) (base + 20));
        float4 r3;
        r3.x = r3_mix.x * comb_scale + r3_base.x;
        r3.y = r3_mix.y * comb_scale + r3_base.y;
        r3.z = r3_mix.z * comb_scale + r3_base.z;
        r3.w = r3_mix.w * comb_scale + r3_base.w;

        const float m0 = fmaxf(fmaxf(r0.x, r0.y), fmaxf(r0.z, r0.w));
        const float m1 = fmaxf(fmaxf(r1.x, r1.y), fmaxf(r1.z, r1.w));
        const float m2 = fmaxf(fmaxf(r2.x, r2.y), fmaxf(r2.z, r2.w));
        const float m3 = fmaxf(fmaxf(r3.x, r3.y), fmaxf(r3.z, r3.w));

        r0.x = expf(r0.x - m0); r0.y = expf(r0.y - m0); r0.z = expf(r0.z - m0); r0.w = expf(r0.w - m0);
        r1.x = expf(r1.x - m1); r1.y = expf(r1.y - m1); r1.z = expf(r1.z - m1); r1.w = expf(r1.w - m1);
        r2.x = expf(r2.x - m2); r2.y = expf(r2.y - m2); r2.z = expf(r2.z - m2); r2.w = expf(r2.w - m2);
        r3.x = expf(r3.x - m3); r3.y = expf(r3.y - m3); r3.z = expf(r3.z - m3); r3.w = expf(r3.w - m3);

        const float r0_sum = 1.0f / (r0.x + r0.y + r0.z + r0.w);
        r0.x = r0.x * r0_sum + epsv; r0.y = r0.y * r0_sum + epsv; r0.z = r0.z * r0_sum + epsv; r0.w = r0.w * r0_sum + epsv;
        
        const float r1_sum = 1.0f / (r1.x + r1.y + r1.z + r1.w);
        r1.x = r1.x * r1_sum + epsv; r1.y = r1.y * r1_sum + epsv; r1.z = r1.z * r1_sum + epsv; r1.w = r1.w * r1_sum + epsv;
        
        const float r2_sum = 1.0f / (r2.x + r2.y + r2.z + r2.w);
        r2.x = r2.x * r2_sum + epsv; r2.y = r2.y * r2_sum + epsv; r2.z = r2.z * r2_sum + epsv; r2.w = r2.w * r2_sum + epsv;
        
        const float r3_sum = 1.0f / (r3.x + r3.y + r3.z + r3.w);
        r3.x = r3.x * r3_sum + epsv; r3.y = r3.y * r3_sum + epsv; r3.z = r3.z * r3_sum + epsv; r3.w = r3.w * r3_sum + epsv;

        float4 col_inv;
        col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
        col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
        col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
        col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

        r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
        r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
        r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
        r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            const float r0_sum_it = 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r0.x *= r0_sum_it; r0.y *= r0_sum_it; r0.z *= r0_sum_it; r0.w *= r0_sum_it;
            
            const float r1_sum_it = 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r1.x *= r1_sum_it; r1.y *= r1_sum_it; r1.z *= r1_sum_it; r1.w *= r1_sum_it;
            
            const float r2_sum_it = 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r2.x *= r2_sum_it; r2.y *= r2_sum_it; r2.z *= r2_sum_it; r2.w *= r2_sum_it;
            
            const float r3_sum_it = 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            r3.x *= r3_sum_it; r3.y *= r3_sum_it; r3.z *= r3_sum_it; r3.w *= r3_sum_it;

            col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
            col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
            col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
            col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

            r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
            r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
            r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
            r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;
        }

        *((float4 *) (out +  8)) = r0;
        *((float4 *) (out + 12)) = r1;
        *((float4 *) (out + 16)) = r2;
        *((float4 *) (out + 20)) = r3;
    }

    __syncthreads();

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

extern "C" __global__ void kernel_dsv4_hc_split_weighted_sum_norm4(
        const struct ds4_hip_args_dsv4_hc_split_weighted_sum_norm args,
        const char  * mixes,
        const float * scale,
        const float * base,
        const char  * x,
        char  * split,
        char  * dst,
        const char  * norm_weight,
        char  * norm_dst) {
    extern __shared__ float shared[];
    uint row = blockIdx.x;
    ushort tid = threadIdx.x;
    ushort sgitg = threadIdx.x / WAVEFRONT_SIZE;
    ushort tiisg = threadIdx.x % WAVEFRONT_SIZE;
    ushort ntg = blockDim.x;

    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || args.n_embd != 4096) {
        return;
    }

    float4 *row_shmem = (float4 *)shared;
    float *pre_shmem = shared + 4096;
    float *sum_shmem = pre_shmem + 4;

    const float *mix = (const float *)(mixes + (uint64_t)row * args.nb_mix1);
    float *out = (float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z_mix  = *((const float4 *) mix);
        const float4 pre_z_base = *((const float4 *) base);
        float4 pre_z;
        pre_z.x = pre_z_mix.x * pre_scale + pre_z_base.x;
        pre_z.y = pre_z_mix.y * pre_scale + pre_z_base.y;
        pre_z.z = pre_z_mix.z * pre_scale + pre_z_base.z;
        pre_z.w = pre_z_mix.w * pre_scale + pre_z_base.w;

        float4 pre;
        pre.x = 1.0f / (1.0f + expf(-pre_z.x)) + epsv;
        pre.y = 1.0f / (1.0f + expf(-pre_z.y)) + epsv;
        pre.z = 1.0f / (1.0f + expf(-pre_z.z)) + epsv;
        pre.w = 1.0f / (1.0f + expf(-pre_z.w)) + epsv;
        *((float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z_mix  = *((const float4 *) (mix  + 4));
        const float4 post_z_base = *((const float4 *) (base + 4));
        float4 post_z;
        post_z.x = post_z_mix.x * post_scale + post_z_base.x;
        post_z.y = post_z_mix.y * post_scale + post_z_base.y;
        post_z.z = post_z_mix.z * post_scale + post_z_base.z;
        post_z.w = post_z_mix.w * post_scale + post_z_base.w;

        float4 post_out;
        post_out.x = 2.0f / (1.0f + expf(-post_z.x));
        post_out.y = 2.0f / (1.0f + expf(-post_z.y));
        post_out.z = 2.0f / (1.0f + expf(-post_z.z));
        post_out.w = 2.0f / (1.0f + expf(-post_z.w));
        *((float4 *)(out + 4)) = post_out;

        const float4 r0_mix  = *((const float4 *) (mix  +  8));
        const float4 r0_base = *((const float4 *) (base +  8));
        float4 r0;
        r0.x = r0_mix.x * comb_scale + r0_base.x;
        r0.y = r0_mix.y * comb_scale + r0_base.y;
        r0.z = r0_mix.z * comb_scale + r0_base.z;
        r0.w = r0_mix.w * comb_scale + r0_base.w;

        const float4 r1_mix  = *((const float4 *) (mix  + 12));
        const float4 r1_base = *((const float4 *) (base + 12));
        float4 r1;
        r1.x = r1_mix.x * comb_scale + r1_base.x;
        r1.y = r1_mix.y * comb_scale + r1_base.y;
        r1.z = r1_mix.z * comb_scale + r1_base.z;
        r1.w = r1_mix.w * comb_scale + r1_base.w;

        const float4 r2_mix  = *((const float4 *) (mix  + 16));
        const float4 r2_base = *((const float4 *) (base + 16));
        float4 r2;
        r2.x = r2_mix.x * comb_scale + r2_base.x;
        r2.y = r2_mix.y * comb_scale + r2_base.y;
        r2.z = r2_mix.z * comb_scale + r2_base.z;
        r2.w = r2_mix.w * comb_scale + r2_base.w;

        const float4 r3_mix  = *((const float4 *) (mix  + 20));
        const float4 r3_base = *((const float4 *) (base + 20));
        float4 r3;
        r3.x = r3_mix.x * comb_scale + r3_base.x;
        r3.y = r3_mix.y * comb_scale + r3_base.y;
        r3.z = r3_mix.z * comb_scale + r3_base.z;
        r3.w = r3_mix.w * comb_scale + r3_base.w;

        const float m0 = fmaxf(fmaxf(r0.x, r0.y), fmaxf(r0.z, r0.w));
        const float m1 = fmaxf(fmaxf(r1.x, r1.y), fmaxf(r1.z, r1.w));
        const float m2 = fmaxf(fmaxf(r2.x, r2.y), fmaxf(r2.z, r2.w));
        const float m3 = fmaxf(fmaxf(r3.x, r3.y), fmaxf(r3.z, r3.w));

        r0.x = expf(r0.x - m0); r0.y = expf(r0.y - m0); r0.z = expf(r0.z - m0); r0.w = expf(r0.w - m0);
        r1.x = expf(r1.x - m1); r1.y = expf(r1.y - m1); r1.z = expf(r1.z - m1); r1.w = expf(r1.w - m1);
        r2.x = expf(r2.x - m2); r2.y = expf(r2.y - m2); r2.z = expf(r2.z - m2); r2.w = expf(r2.w - m2);
        r3.x = expf(r3.x - m3); r3.y = expf(r3.y - m3); r3.z = expf(r3.z - m3); r3.w = expf(r3.w - m3);

        const float r0_sum = 1.0f / (r0.x + r0.y + r0.z + r0.w);
        r0.x = r0.x * r0_sum + epsv; r0.y = r0.y * r0_sum + epsv; r0.z = r0.z * r0_sum + epsv; r0.w = r0.w * r0_sum + epsv;
        
        const float r1_sum = 1.0f / (r1.x + r1.y + r1.z + r1.w);
        r1.x = r1.x * r1_sum + epsv; r1.y = r1.y * r1_sum + epsv; r1.z = r1.z * r1_sum + epsv; r1.w = r1.w * r1_sum + epsv;
        
        const float r2_sum = 1.0f / (r2.x + r2.y + r2.z + r2.w);
        r2.x = r2.x * r2_sum + epsv; r2.y = r2.y * r2_sum + epsv; r2.z = r2.z * r2_sum + epsv; r2.w = r2.w * r2_sum + epsv;
        
        const float r3_sum = 1.0f / (r3.x + r3.y + r3.z + r3.w);
        r3.x = r3.x * r3_sum + epsv; r3.y = r3.y * r3_sum + epsv; r3.z = r3.z * r3_sum + epsv; r3.w = r3.w * r3_sum + epsv;

        float4 col_inv;
        col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
        col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
        col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
        col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

        r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
        r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
        r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
        r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            const float r0_sum_it = 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r0.x *= r0_sum_it; r0.y *= r0_sum_it; r0.z *= r0_sum_it; r0.w *= r0_sum_it;
            
            const float r1_sum_it = 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r1.x *= r1_sum_it; r1.y *= r1_sum_it; r1.z *= r1_sum_it; r1.w *= r1_sum_it;
            
            const float r2_sum_it = 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r2.x *= r2_sum_it; r2.y *= r2_sum_it; r2.z *= r2_sum_it; r2.w *= r2_sum_it;
            
            const float r3_sum_it = 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            r3.x *= r3_sum_it; r3.y *= r3_sum_it; r3.z *= r3_sum_it; r3.w *= r3_sum_it;

            col_inv.x = 1.0f / (r0.x + r1.x + r2.x + r3.x + epsv);
            col_inv.y = 1.0f / (r0.y + r1.y + r2.y + r3.y + epsv);
            col_inv.z = 1.0f / (r0.z + r1.z + r2.z + r3.z + epsv);
            col_inv.w = 1.0f / (r0.w + r1.w + r2.w + r3.w + epsv);

            r0.x *= col_inv.x; r0.y *= col_inv.y; r0.z *= col_inv.z; r0.w *= col_inv.w;
            r1.x *= col_inv.x; r1.y *= col_inv.y; r1.z *= col_inv.z; r1.w *= col_inv.w;
            r2.x *= col_inv.x; r2.y *= col_inv.y; r2.z *= col_inv.z; r2.w *= col_inv.w;
            r3.x *= col_inv.x; r3.y *= col_inv.y; r3.z *= col_inv.z; r3.w *= col_inv.w;
        }

        *((float4 *)(out + 8)) = r0;
        *((float4 *)(out + 12)) = r1;
        *((float4 *)(out + 16)) = r2;
        *((float4 *)(out + 20)) = r3;
    }

    __syncthreads();

    float sumf = 0.0f;
    const uint n4 = 1024u;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 *x0 = (const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        const float4 *x1 = (const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        const float4 *x2 = (const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        const float4 *x3 = (const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        
        float4 v0 = x0[i];
        float4 v1 = x1[i];
        float4 v2 = x2[i];
        float4 v3 = x3[i];

        float4 v;
        v.x = v0.x * pre_shmem[0] + v1.x * pre_shmem[1] + v2.x * pre_shmem[2] + v3.x * pre_shmem[3];
        v.y = v0.y * pre_shmem[0] + v1.y * pre_shmem[1] + v2.y * pre_shmem[2] + v3.y * pre_shmem[3];
        v.z = v0.z * pre_shmem[0] + v1.z * pre_shmem[1] + v2.z * pre_shmem[2] + v3.z * pre_shmem[3];
        v.w = v0.w * pre_shmem[0] + v1.w * pre_shmem[1] + v2.w * pre_shmem[2] + v3.w * pre_shmem[3];

        row_shmem[i] = v;
        sumf += dot_f4(v, v);
    }

    sumf = warp_reduce_sum(sumf);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    __syncthreads();

    sumf = (tiisg < (ntg / WAVEFRONT_SIZE) ? sum_shmem[tiisg] : 0.0f);
    sumf = warp_reduce_sum(sumf);
    
    const float norm_scale = rsqrtf(sumf / 4096.0f + args.norm_eps);

    float4 *dst4 = (float4 *)(dst + (uint64_t)row * args.nb1);
    const float4 *w4 = (const float4 *)norm_weight;
    float4 *norm4 = (float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        float4 v = row_shmem[i];
        dst4[i] = v;
        
        float4 nw = w4[i];
        float4 res;
        res.x = (v.x * norm_scale) * nw.x;
        res.y = (v.y * norm_scale) * nw.y;
        res.z = (v.z * norm_scale) * nw.z;
        res.w = (v.w * norm_scale) * nw.w;
        norm4[i] = res;
    }
}

extern "C" __global__ void kernel_dsv4_hc_expand(
        const struct ds4_hip_args_dsv4_hc_expand args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        const char * block_add,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

extern "C" __global__ void kernel_dsv4_hc_expand4(
        const struct ds4_hip_args_dsv4_hc_expand args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        const char * block_add,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

extern "C" __global__ void kernel_dsv4_hc_weighted_sum(
        const struct ds4_hip_args_dsv4_hc_weighted_sum args,
        const char * x,
        const char * weights,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}

extern "C" __global__ void kernel_dsv4_shared_down_hc_expand4_q8_0(
        const struct ds4_hip_args_mul_mv mv,
        const struct ds4_hip_args_dsv4_hc_expand hc,
        const char * weight,
        const char * shared_mid,
        char * shared_out,
        const char * routed_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    extern __shared__ char shmem[];
    
    ushort tiisg = threadIdx.x % WAVEFRONT_SIZE;
    ushort sgitg = threadIdx.x / WAVEFRONT_SIZE;

    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const short NQ = 8;
    const short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = blockIdx.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    const float *y = (const float *)(shared_mid);
    const float *yb = y + ib0 * QK8_0 + il * NQ;

    const block_q8_0 *ax[N_R0_Q8_0];
    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (const block_q8_0 *)(weight + off0);
    }

    float sumf[N_R0_Q8_0] = { 0.0f };
    float yl[8];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma unroll
        for(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        #pragma unroll
        for(short row = 0; row < N_R0_Q8_0; ++row) {
            const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            #pragma unroll
            for(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    float *shmem_f32[N_R0_Q8_0];
    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        shmem_f32[row] = (float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = warp_reduce_sum(sumf[row]);
    }

    __syncthreads();

    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    __syncthreads();

    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = warp_reduce_sum(tiisg < (blockDim.x / WAVEFRONT_SIZE) ? shmem_f32[row][tiisg] : 0.0f);
        if (tiisg == 0 && sgitg == 0) {
            *((float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((const float *)(post + dst_hc * hc.nb_post0));

                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

extern "C" __global__ void kernel_dsv4_q8_hc_expand4_q8_0(
        const struct ds4_hip_args_mul_mv mv,
        const struct ds4_hip_args_dsv4_hc_expand hc,
        const char * weight,
        const char * input,
        char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    extern __shared__ char shmem[];
    ushort tiisg = threadIdx.x % WAVEFRONT_SIZE;
    ushort sgitg = threadIdx.x / WAVEFRONT_SIZE;

    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    const short NW = N_SIMDWIDTH;
    const short NQ = 8;
    const short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = blockIdx.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    const float *y = (const float *)(input);
    const float *yb = y + ib0 * QK8_0 + il * NQ;

    const block_q8_0 *ax[N_R0_Q8_0];
    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (const block_q8_0 *)(weight + off0);
    }

    float sumf[N_R0_Q8_0] = { 0.0f };
    float yl[8];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        #pragma unroll
        for(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        #pragma unroll
        for(short row = 0; row < N_R0_Q8_0; ++row) {
            const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            #pragma unroll
            for(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    float *shmem_f32[N_R0_Q8_0];
    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        shmem_f32[row] = (float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = warp_reduce_sum(sumf[row]);
    }

    __syncthreads();

    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    __syncthreads();

    #pragma unroll
    for(short row = 0; row < N_R0_Q8_0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = warp_reduce_sum(tiisg < (blockDim.x / WAVEFRONT_SIZE) ? shmem_f32[row][tiisg] : 0.0f);
        if (tiisg == 0 && sgitg == 0) {
            *((float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((const float *)(post + dst_hc * hc.nb_post0));

                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

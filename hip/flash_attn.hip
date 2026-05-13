#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>
#include <float.h>
#include "../ds4_gpu_common.h"

#define WAVEFRONT_SIZE 32
#define N_SIMDWIDTH 32

#define MAX_HALF 65504.0f

#define FC_FLASH_ATTN_EXT_PAD 100
#define FC_FLASH_ATTN_EXT_BLK 200
#define FC_FLASH_ATTN_EXT 300
#define FC_FLASH_ATTN_EXT_VEC 400
#define FC_FLASH_ATTN_EXT_VEC_REDUCE 500
#define OP_FLASH_ATTN_EXT_NQPSG 8
#define OP_FLASH_ATTN_EXT_NCPSG 64
#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32

#ifndef PAD2
#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#endif

// WMMA types and builtins for GFX11 (Wave32)
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

struct half4 {
    half x, y, z, w;
};

#define FOR_UNROLL _Pragma("unroll")

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    return val;
}

__device__ inline float warp_allreduce_sum(float val) {
    return warp_reduce_sum(val);
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor(val, offset, WAVEFRONT_SIZE));
    return val;
}

__device__ inline float warp_allreduce_max(float val) {
    return warp_reduce_max(val);
}

__device__ inline float dot_f4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ inline float dot_h4(half4 a, half4 b) {
    return (float)a.x * (float)b.x + (float)a.y * (float)b.y + (float)a.z * (float)b.z + (float)a.w * (float)b.w;
}

template <typename T4>
__device__ void dequantize_f32_t4(const float4 * src, short il, T4 & reg) {
    reg = *src;
}

__device__ void dequantize_f16_t4(const half4 * src, short il, __half & reg) {
    reg = ((const __half*)src)[il];
}

__device__ void dequantize_f16_t4(const half4 * src, short il, half4 & reg) {
    reg = src[il];
}

__device__ void dequantize_f16(const half4 * src, short il, half4 & reg) {
    reg = src[il];
}

// WMMA-accelerated 8x8 block multiply-accumulate.
// This function leverages the 16x16x16 WMMA hardware instruction on GFX11 Wave32.
// It maps the 8x8 problem into a 16x16 space with zero padding.
template<typename T>
__device__ inline void matmul8x8_block(
        float mqk[2], 
        const T * pq, int ldq,
        const __half * pk, int ldk,
        bool transpose_k) {
    
    const short t = threadIdx.x % WAVEFRONT_SIZE;
    const short row = t % 16;
    const short col_off = (t / 16) * 8;

    // Load A matrix (Q or Attention probabilities)
    // Row-major [16, 16] where we use [8, 8] part.
    half16 a;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k < 8 && row < 8 && col_off + k < 8) {
            a[k] = (__half)pq[row * ldq + col_off + k];
        } else {
            a[k] = (__half)0.0f;
        }
    }

    // Load B matrix (K or V)
    // Hardware WMMA expects B to be column-major.
    // If transpose_k is true (Q*K^T), we load K as row-major, so the hardware sees K^T (col-major).
    // If transpose_k is false (S*V), we must load V as column-major (strided loads).
    half16 b;
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        if (k < 8) {
            if (transpose_k) {
                b[k] = (row < 8 && col_off + k < 8) ? pk[row * ldk + col_off + k] : (__half)0.0f;
            } else {
                b[k] = (col_off + k < 8 && row < 8) ? pk[(col_off + k) * ldk + row] : (__half)0.0f;
            }
        } else {
            b[k] = (__half)0.0f;
        }
    }

    float8 c;
    #pragma unroll
    for (int k = 0; k < 8; ++k) c[k] = 0.0f;

    // Execute hardware matrix multiplication
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);

    // Shuffle results back to the expected thread-local layout for Flash Attention.
    // Original layout: mqk[0] = C[t/8, t%8], mqk[1] = C[t/8 + 4, t%8]
    // WMMA layout: thread t has row t%16 and columns (t/16)*8 + 0..7
    const short col = t % 8;
    mqk[0] += __shfl(c[col], t / 8, 32);
    mqk[1] += __shfl(c[col], t / 8 + 4, 32);
}

extern "C" __global__ void kernel_flash_attn_ext_pad(
        struct ds4_hip_args_flash_attn_ext_pad args,
        const char * k,
        const char * v,
        const char * mask,
        char * dst) {
    const int32_t C = args.ncpsg;

    char * k_pad    = dst;
    char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).x;
    const int32_t i2 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).y;
    const int32_t i3 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).z;

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            for (uint64_t i = threadIdx.x; i < args.nb11; i += blockDim.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = threadIdx.x; i < args.nb21; i += blockDim.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = threadIdx.x; i < args.nb11; i += blockDim.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = threadIdx.x; i < args.nb21; i += blockDim.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (args.has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                const __half * mask_src = (const __half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                __half * mask_dst = (__half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = threadIdx.x; i < C; i += blockDim.x) {
                    if (i >= icp) {
                        mask_dst[i] = __float2half(-MAX_HALF);
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

extern "C" __global__ void kernel_flash_attn_ext_blk(
        struct ds4_hip_args_flash_attn_ext_blk args,
        const char * mask,
        char * dst) {
    const int32_t Q = args.nqptg;
    const int32_t C = args.ncpsg;

    const short tiisg = threadIdx.x % WAVEFRONT_SIZE;

    const int32_t i3 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).z/args.ne32;
    const int32_t i2 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).z%args.ne32;
    const int32_t i1 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).y;
    const int32_t i0 = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).x;

    char res = i0*C + C > args.ne30 ? 1 : 0;

    const __half * mask_src = (const __half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    if ((C > N_SIMDWIDTH || Q > 1) && res == 0) {
        float mmin =  MAX_HALF;
        float mmax = -MAX_HALF;

        for (short j = 0; j < Q; ++j) {
            for (short ii = 0; ii < C/N_SIMDWIDTH; ++ii) {
                float val = (float)mask_src[ii*N_SIMDWIDTH];
                mmin = fminf(mmin, val);
                mmax = fmaxf(mmax, val);
            }

            mask_src += args.nb31/2;
        }

        mmin = warp_reduce_max(-mmin);
        mmin = -mmin;
        mmax = warp_reduce_max(mmax);

        if (mmax > -MAX_HALF) {
            if (mmin == 0.0f && mmax == 0.0f) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}

template<
    typename q_t,
    typename q4_t,
    typename k_t,
    typename k4x4_t,
    typename v_t,
    typename v4x4_t,
    typename kd4x4_t,
    short nl_k,
    typename vd4x4_t,
    short nl_v,
    short DK,
    short DV,
    short Q,
    short C,
    short NSG>
__device__ void kernel_flash_attn_ext_impl(
        struct ds4_hip_args_flash_attn_ext args,
        const char * q,
        const char * k,
        const char * v,
        const char * mask,
        const char * sinks,
        const char * pad,
        const char * blk,
        char * dst,
        __half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    const ushort iq3 = tgpig.z;
    const ushort iq2 = tgpig.y;
    const ushort iq1 = tgpig.x*Q;

    const int NS10 = args.ns10;
    const int NS20 = args.ns20;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C;

    constexpr short T  = DK + 2*PV;

    __half  * sq  = shmem_f16 + 0*T;
    float4 * sq4 = (float4 *) (shmem_f16 + 0*T);
    float4 * so4 = (float4 *) (shmem_f16 + 0*T + Q*DK);
    float   * ss  = (float   *) (shmem_f16 + Q*T);
    float2  * ss2 = (float2  *) (shmem_f16 + Q*T);

    __half * sk = shmem_f16 + sgitg*(4*16*8) + Q*T + Q*(2*SH);

    float2 * sm2 = (float2 *) (shmem_f16 + Q*T + 2*C);

    const float2 * pm2[NQ];

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        pm2[jj] = (const float2 *) ((const char *) mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
        const int32_t nblk0 = ((args.ne11 + C - 1)/C);
        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;
        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);
        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        const float4 * q4_ptr = (const float4 *) ((const char *) q + j*args.nb01);
        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = q4_ptr[i];
            } else {
                sq4[j*DK4 + i] = make_float4(0,0,0,0);
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        for (short i = tiisg; i < PV4; i += NW) {
            so4[j*PV4 + i] = make_float4(0,0,0,0);
        }
        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    __syncthreads();

    float S[NQ];
    #pragma unroll
    for(int i=0; i<NQ; i++) S[i] = 0.0f;

    {
        float M[NQ];
        #pragma unroll
        for(int i=0; i<NQ; i++) M[i] = -FLT_MAX/2;

        float slope = 1.0f;
        if (args.has_bias) {
            const short h = iq2;
            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
            slope = powf(base, (float)exph);
        }

        for (int ic0 = 0; ; ++ic0) {
            int ic = ic0*C;
            if (ic >= args.ne11) break;

            if (args.has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);
                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!args.has_mask) {
                    __half * sm = (__half *) (sm2);
                    for (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;
                        for (short i = tiisg; i < C; i += NW) {
                            if (ic + i >= args.ne11) sm[2*j*SH + i] = __float2half(-MAX_HALF);
                        }
                    }
                } else {
                    for (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;
                        pm2[jj] = (const float2 *) ((const __half *) mask +
                                (iq1 + j)*C +
                                (iq2%args.ne32)*(C*args.ne31) +
                                (iq3%args.ne33)*(C*args.ne31*args.ne32));
                    }
                }
                ic = 0;
            }

            char blk_cur = 1;
            if (args.has_mask) {
                blk_cur = blk[ic0];
                if (blk_cur == 0) {
                    for (short jj = 0; jj < NQ; ++jj) pm2[jj] += NW;
                    continue;
                }
                if (blk_cur == 1) {
                    for (short jj = 0; jj < NQ; ++jj) {
                        const short j = jj*NSG + sgitg;
                        if (args.bc_mask) {
                            sm2[j*SH + tiisg] = (iq1 + j) < args.ne31 ? pm2[jj][tiisg] : make_float2(-MAX_HALF, -MAX_HALF);
                        } else {
                            sm2[j*SH + tiisg] = pm2[jj][tiisg];
                        }
                        pm2[jj] += NW;
                    }
                } else if (blk_cur == 2) {
                    for (short jj = 0; jj < NQ; ++jj) pm2[jj] += NW;
                }
            }

            if (sizeof(kd4x4_t) == sizeof(k4x4_t)) {
                const __half * pk_ptr = (const __half *) (k + ic*args.nb11);
                __half * pq_ptr = sq;
                float * ps_ptr = ss;

                pk_ptr += sgitg*(8*NS10);
                ps_ptr += sgitg*(8*1);

                constexpr short NC = (C/8)/NSG;

                for (short cc = 0; cc < NC; ++cc) {
                    float mqk[2] = {0, 0};
                    for (short i = 0; i < DK8; ++i) {
                        matmul8x8_block(mqk, pq_ptr + 8*i, DK, pk_ptr + 8*i, NS10, true);
                    }
                    const short t = tiisg;
                    ps_ptr[(t/8)*SH + (t%8)] = mqk[0];
                    ps_ptr[(t/8+4)*SH + (t%8)] = mqk[1];

                    pk_ptr += 8*(NSG*NS10);
                    ps_ptr += 8*(NSG);
                }
            } else {
                for (short ccc = 0; ccc < (C/8)/NSG; ++ccc) {
                    const short cc = ccc*NSG + sgitg;
                    const short tx = tiisg%4;
                    const short ty = tiisg/4;

                    float mqk[2] = {0, 0};

                    for (short ii = 0; ii < DK16; ii += 4) {
                        const kd4x4_t * pk4x4 = (const kd4x4_t *) (k + ((ic + 8*cc + ty)*args.nb11));
                        if (ii + tx < DK16) {
                            k4x4_t tmp;
                            dequantize_f16_t4((const half4 *)(pk4x4 + (ii + tx)/nl_k), (ii + tx)%nl_k, tmp);
                            ((half4 *)sk)[4*ty + tx] = *((half4*)&tmp);
                        }
                        __syncthreads();

                        for (short k_idx = 0; k_idx < 4 && ii + k_idx < DK16; ++k_idx) {
                            matmul8x8_block(mqk, sq + (2*(ii + k_idx))*8, DK, sk + 16*k_idx, 4*16, true);
                            matmul8x8_block(mqk, sq + (2*(ii + k_idx) + 1)*8, DK, sk + 16*k_idx + 8, 4*16, true);
                        }
                        __syncthreads();
                    }
                    const short t = tiisg;
                    ss[(t/8)*SH + (t%8) + 8*cc] = mqk[0];
                    ss[(t/8+4)*SH + (t%8) + 8*cc] = mqk[1];
                }
            }

            __syncthreads();

            for (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;
                const float m = M[jj];
                float2 s2 = ss2[j*SH/2 + tiisg];
                s2.x *= args.scale; s2.y *= args.scale;

                if (args.has_scap) {
                    s2.x = args.logit_softcap * tanhf(s2.x);
                    s2.y = args.logit_softcap * tanhf(s2.y);
                }

                if (blk_cur != 2) {
                    float2 mask_val = sm2[j*SH + tiisg];
                    if (args.has_bias) {
                        s2.x += mask_val.x * slope;
                        s2.y += mask_val.y * slope;
                    } else {
                        s2.x += mask_val.x;
                        s2.y += mask_val.y;
                    }
                }

                M[jj] = warp_allreduce_max(fmaxf(M[jj], fmaxf(s2.x, s2.y)));
                const float ms = expf(m - M[jj]);
                const float2 vs2 = make_float2(expf(s2.x - M[jj]), expf(s2.y - M[jj]));

                S[jj] = S[jj]*ms + warp_reduce_sum(vs2.x + vs2.y);
                ss2[j*SH/2 + tiisg] = vs2;

                for (short i = tiisg; i < DV4; i += NW) {
                    float4 val = so4[j*PV4 + i];
                    val.x *= ms; val.y *= ms; val.z *= ms; val.w *= ms;
                    so4[j*PV4 + i] = val;
                }
            }

            __syncthreads();

            if (sizeof(vd4x4_t) == sizeof(v4x4_t)) {
                constexpr short NO = PV4/NSG;
                const k_t * pv_ptr = (const k_t *) (v + ic*args.nb21);
                pv_ptr += 8*sgitg;

                for (short cc = 0; cc < C/8; ++cc) {
                    for (short ii = 0; ii < NO; ++ii) {
                        float mqk[2] = {0, 0};
                        matmul8x8_block(mqk, (const float *)(ss + 8*cc), SH, (const __half*)(pv_ptr + 16*ii*NSG), NS20, false);
                        
                        float4 val = so4[sgitg*PV4 + 2*ii]; 
                        val.x += mqk[0]; 
                        so4[sgitg*PV4 + 2*ii] = val;
                    }
                    pv_ptr += 8*NS20;
                }
            }

            __syncthreads();
        }

        if (args.has_sinks) {
            for (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;
                const float m = M[jj];
                const float s = tiisg == 0 ? ((const float *) sinks)[iq2] : -FLT_MAX/2;

                M[jj] = warp_allreduce_max(fmaxf(M[jj], s));
                const float ms = expf(m - M[jj]);
                const float vs = expf(s - M[jj]);

                S[jj] = S[jj]*ms + warp_reduce_sum(vs);
                for (short i = tiisg; i < DV4; i += NW) {
                    float4 val = so4[j*PV4 + i];
                    val.x *= ms; val.y *= ms; val.z *= ms; val.w *= ms;
                    so4[j*PV4 + i] = val;
                }
            }
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) break;

        float4 * dst4 = (float4 *) dst + ((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4;
        const float scale = S[jj] == 0.0f ? 0.0f : 1.0f/S[jj];

        for (short i = tiisg; i < DV4; i += NW) {
            float4 val = so4[j*PV4 + i];
            val.x *= scale; val.y *= scale; val.z *= scale; val.w *= scale;
            dst4[i] = val;
        }
    }
}

extern "C" __global__ void kernel_flash_attn_ext_f16_dk512_dv512(
        struct ds4_hip_args_flash_attn_ext args,
        const char * q,
        const char * k,
        const char * v,
        const char * mask,
        const char * sinks,
        const char * pad,
        const char * blk,
        char * dst) {
    
    extern __shared__ __half shmem[];
    ushort tiisg = threadIdx.x % WAVEFRONT_SIZE;
    ushort sgitg = threadIdx.x / WAVEFRONT_SIZE;

    if (args.nsg == 4)
        kernel_flash_attn_ext_impl<__half, float4, __half, __half, __half, __half, __half, 1, __half, 1, 512, 512, 8, 64, 4>
            (args, q, k, v, mask, sinks, pad, blk, dst, shmem, make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), tiisg, sgitg);
    else if (args.nsg == 8)
        kernel_flash_attn_ext_impl<__half, float4, __half, __half, __half, __half, __half, 1, __half, 1, 512, 512, 8, 64, 8>
            (args, q, k, v, mask, sinks, pad, blk, dst, shmem, make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), tiisg, sgitg);
}

template<
    typename q4_t,
    typename k4_t,
    typename v4_t,
    typename qk_t,
    typename s_t,
    typename s4_t,
    typename o4_t,
    typename kd4_t,
    short nl_k,
    typename vd4_t,
    short nl_v,
    short DK,
    short DV,
    short NE = 4,
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>
__device__ void kernel_flash_attn_ext_vec_impl(
        struct ds4_hip_args_flash_attn_ext_vec args,
        const char * q,
        const char * k,
        const char * v,
        const char * mask,
        const char * sinks,
        const char * pad,
        char * dst,
        __half * shmem_f16,
        uint3   tgpig,
        ushort  tiisg,
        ushort  sgitg) {
    
    const int NWG = args.nwg;
    const int NSG = args.nsg;
    const int NS10 = args.ns10;
    const int NS20 = args.ns20;

    const short iwg = tgpig.z % NWG;
    const ushort iq3 = tgpig.z / NWG;
    const ushort iq2 = tgpig.y;
    const ushort iq1 = tgpig.x;

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    float4 * sq4 = (float4 *) (shmem_f16 + 0*PK);
    float  * ss  = (float  *) (shmem_f16 + sgitg*SH + NSG*PK);
    float4 * ss4 = (float4 *) (shmem_f16 + sgitg*SH + NSG*PK);
    __half  * sm  = (__half  *) (shmem_f16 + sgitg*SH + 2*C + NSG*PK);
    float4 * so4 = (float4 *) (shmem_f16 + 2*sgitg*PV + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;
        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);
        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    const float4 * q4_ptr = (const float4 *) q;
    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) sq4[i] = q4_ptr[i];
            else sq4[i] = make_float4(0,0,0,0);
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = make_float4(0,0,0,0);
    }
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = make_float4(0,0,0,0);
    }

    __syncthreads();

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        const __half * pm = (const __half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);

        float slope = 1.0f;
        if (args.has_bias) {
            const short h = iq2;
            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
            slope = powf(base, (float)exph);
        }

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) break;

            if (args.has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;
                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);
                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!args.has_mask) {
                    if (ic + tiisg >= args.ne11) sm[tiisg] = __float2half(-MAX_HALF);
                } else {
                    pm = (const __half *) mask + iq1*C + (iq2%args.ne32)*(C*args.ne31) + (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }
                ic = 0;
            }

            if (args.has_mask) sm[tiisg] = pm[ic + tiisg];

            if (warp_allreduce_max((float)sm[tiisg]) <= -MAX_HALF) continue;

            {
                const float4 * pk4 = (const float4 *) (k + ic*args.nb11);
                pk4 += ty*NS10/4 + tx;

                float mqk[C/NE];
                for (short cc = 0; cc < C/NE; ++cc) {
                    mqk[cc] = 0.0f;
                    for (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[cc] += dot_f4(pk4[cc*NE*NS10/4 + ii*NL], sq4[ii*NL + tx]); 
                    }

                    if (NE == 1) {
                        mqk[cc] = warp_allreduce_sum(mqk[cc]);
                    } else {
                        for (int offset = NE / 2; offset > 0; offset /= 2)
                            mqk[cc] += __shfl_down(mqk[cc], offset, NW);
                        mqk[cc] = __shfl(mqk[cc], NL*ty, NW);
                    }
                }

                mqk[tx] *= args.scale;
                if (args.has_scap) mqk[tx] = args.logit_softcap * tanhf(mqk[tx]);
                if (args.has_bias) mqk[tx] += (float)sm[NE*tx + ty] * slope;
                else mqk[tx] += (float)sm[NE*tx + ty];

                ss[NE*tx + ty] = mqk[tx];
            }

            __syncthreads();

            {
                const float m = M;
                const float s = ss[tiisg];
                M = warp_allreduce_max(fmaxf(M, s));
                const float ms = expf(m - M);
                const float vs = expf(s - M);
                S = S*ms + warp_reduce_sum(vs);
                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    for (short ii = 0; ii < DV4/NL; ++ii) {
                        float4 val = so4[ii*NL];
                        val.x *= ms; val.y *= ms; val.z *= ms; val.w *= ms;
                        so4[ii*NL] = val;
                    }
                }
            }

            __syncthreads();

            {
                float4 lo[DV4/NL];
                for (short ii = 0; ii < DV4/NL; ++ii) lo[ii] = make_float4(0,0,0,0);

                const float4 * pv4 = (const float4 *) (v + ic*args.nb21);
                pv4 += ty*NS20/4 + tx;
                const float * sst = ss + ty;

                for (short cc = 0; cc < C/NE; ++cc) {
                    for (short ii = 0; ii < DV4/NL; ++ii) {
                        float4 v_val = pv4[cc*NE*NS20/4 + ii*NL];
                        float s_val = sst[cc*NE];
                        lo[ii].x += v_val.x * s_val;
                        lo[ii].y += v_val.y * s_val;
                        lo[ii].z += v_val.z * s_val;
                        lo[ii].w += v_val.w * s_val;
                    }
                }

                for (short ii = 0; ii < DV4/NL; ++ii) {
                    if (NE > 1) {
                        for (int offset = NE / 2; offset > 0; offset /= 2) {
                            lo[ii].x += __shfl_down(lo[ii].x, offset, NW);
                            lo[ii].y += __shfl_down(lo[ii].y, offset, NW);
                            lo[ii].z += __shfl_down(lo[ii].z, offset, NW);
                            lo[ii].w += __shfl_down(lo[ii].w, offset, NW);
                        }
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    for (short ii = 0; ii < DV4/NL; ++ii) {
                        float4 val = so4[ii*NL];
                        val.x += lo[ii].x; val.y += lo[ii].y; val.z += lo[ii].z; val.w += lo[ii].w;
                        so4[ii*NL] = val;
                    }
                }
            }
        }

        if (args.has_sinks && sgitg == 0 && iwg == 0) {
            const float m = M;
            const float s = tiisg == 0 ? ((const float *) sinks)[iq2] : -FLT_MAX/2;
            M = warp_allreduce_max(fmaxf(M, s));
            const float ms = expf(m - M);
            const float vs = expf(s - M);
            S = S*ms + warp_reduce_sum(vs);
            if ((DV4/NL % NW == 0) || ty == 0) {
                for (short ii = 0; ii < DV4/NL; ++ii) {
                    float4 val = so4[ii*NL];
                    val.x *= ms; val.y *= ms; val.z *= ms; val.w *= ms;
                    so4[ii*NL] = val;
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = S;
            ss[1] = M;
        }
    }

    so4 -= tiisg;
    __syncthreads();

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[0];
            const float S1 = ss[r*(SH/2) + 0];
            const float M0 = ss[1];
            const float M1 = ss[r*(SH/2) + 1];
            const float M = fmaxf(M0, M1);
            const float ms0 = expf(M0 - M);
            const float ms1 = expf(M1 - M);
            const float S = S0*ms0 + S1*ms1;
            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }
            for (short i = tiisg; i < DV4; i += NW) {
                float4 v0 = so4[i];
                float4 v1 = so4[i + r*PV4];
                v0.x = v0.x*ms0 + v1.x*ms1;
                v0.y = v0.y*ms0 + v1.y*ms1;
                v0.z = v0.z*ms0 + v1.z*ms1;
                v0.w = v0.w*ms0 + v1.w*ms1;
                so4[i] = v0;
            }
        }
        __syncthreads();
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = (int64_t)iq3*args.ne2*args.ne1 + iq2 + (int64_t)iq1*args.ne1;
        float4 * dst4 = (float4 *) dst;
        float  * dst1 = (float  *) dst + nrows*DV*NWG;
        const float S = NWG == 1 ? (ss[0] == 0.0f ? 0.0f : 1.0f/ss[0]) : 1.0f;

        for (short i = tiisg; i < DV4; i += NW) {
            float4 val = so4[i];
            val.x *= S; val.y *= S; val.z *= S; val.w *= S;
            dst4[rid*DV4*NWG + NWG*i + iwg] = val;
        }
        if (NWG > 1 && tiisg == 0) {
            dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
            dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
        }
    }
}

extern "C" __global__ void kernel_flash_attn_ext_vec_f16_dk512_dv512(
        struct ds4_hip_args_flash_attn_ext_vec args,
        const char * q,
        const char * k,
        const char * v,
        const char * mask,
        const char * sinks,
        const char * pad,
        char * dst) {
    
    extern __shared__ __half shmem[];
    ushort tiisg = threadIdx.x % WAVEFRONT_SIZE;
    ushort sgitg = threadIdx.x / WAVEFRONT_SIZE;

    kernel_flash_attn_ext_vec_impl<float4, float4, float4, float, float, float4, float4, __half, 1, __half, 1, 512, 512, 1>
        (args, q, k, v, mask, sinks, pad, dst, shmem, make_uint3(blockIdx.x, blockIdx.y, blockIdx.z), tiisg, sgitg);
}

extern "C" __global__ void kernel_flash_attn_ext_vec_reduce(
        struct ds4_hip_args_flash_attn_ext_vec_reduce args,
        const char * htmp,
        char * dst) {
    const int NWG = args.nwg;
    const int DV = args.dv;
    const uint64_t rid = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z).x;
    const short iwg = threadIdx.x % WAVEFRONT_SIZE;

    if (iwg >= NWG) return;

    const float * ss = (const float *) htmp + (uint64_t)args.nrows*DV*NWG;
    float S_val = ss[rid*(2*NWG) + 2*iwg + 0];
    float M_val = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = warp_allreduce_max(M_val);
    const float ms = expf(M_val - m);

    float S = warp_allreduce_sum(S_val*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;
    const float4 * htmp4 = (const float4 *) htmp + rid*DV4*NWG;
    float4 * dst4  = (float4 *) dst  + rid*DV4;

    for (short i = threadIdx.x / WAVEFRONT_SIZE; i < DV4; i += blockDim.x / WAVEFRONT_SIZE) {
        float4 v = htmp4[i*NWG + iwg];
        v.x *= ms; v.y *= ms; v.z *= ms; v.w *= ms;
        
        for (int offset = NWG / 2; offset > 0; offset /= 2) {
            v.x += __shfl_down(v.x, offset, NWG);
            v.y += __shfl_down(v.y, offset, NWG);
            v.z += __shfl_down(v.z, offset, NWG);
            v.w += __shfl_down(v.w, offset, NWG);
        }

        if (iwg == 0) {
            v.x *= S; v.y *= S; v.z *= S; v.w *= S;
            dst4[i] = v;
        }
    }
}

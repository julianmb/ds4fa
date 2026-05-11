#include "../ds4_gpu_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ const float GELU_COEF_A     = 0.044715f;
__device__ const float GELU_QUICK_COEF = -1.702f;
__device__ const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
__device__ const float SQRT_2_INV      = 0.70710678118654752440084436210484f;

__device__ const float p_erf  = 0.3275911f;
__device__ const float a1_erf = 0.254829592f;
__device__ const float a2_erf = -0.284496736f;
__device__ const float a3_erf = 1.421413741f;
__device__ const float a4_erf = -1.453152027f;
__device__ const float a5_erf = 1.061405429f;

template<typename T>
__device__ inline T erf_approx(T x) {
    T sign_x = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
    x = fabsf(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * expf(-x * x);
    return sign_x * y;
}

template<typename T> __device__ T elu_approx(T x);

template<> __device__ inline float elu_approx<float>(float x) {
    return (x > 0.f) ? x : (expf(x) - 1.0f);
}

template<> __device__ inline float4 elu_approx<float4>(float4 x) {
    float4 res;
    res.x = (x.x > 0.0f) ? x.x : (expf(x.x) - 1.0f);
    res.y = (x.y > 0.0f) ? x.y : (expf(x.y) - 1.0f);
    res.z = (x.z > 0.0f) ? x.z : (expf(x.z) - 1.0f);
    res.w = (x.w > 0.0f) ? x.w : (expf(x.w) - 1.0f);
    return res;
}

template <typename T0, typename T, typename TC>
__device__ void kernel_unary_impl(
        ds4_hip_args_unary args,
        const char * src0,
        char * dst) {
    const T0 * src0_ptr;
          T  * dst_ptr;

    int i0;

    if (args.cnt) {
        i0 = blockIdx.x;

        src0_ptr = (const T0 *) (src0);
        dst_ptr  = (      T  *) (dst);
    } else {
        const int i03 = blockIdx.z;
        const int i02 = blockIdx.y;
        const int k0  = blockIdx.x / args.ne01;
        const int i01 = blockIdx.x - k0 * args.ne01;

        i0 = k0 * blockDim.x + threadIdx.x;

        src0_ptr = (const T0 *) (src0 + i03 * args.nb03 + i02 * args.nb02 + i01 * args.nb01);
        dst_ptr  = (      T  *) (dst  + i03 * args.nb3  + i02 * args.nb2  + i01 * args.nb1 );
    }

    if (!args.cnt) {
        if (i0 >= args.ne0) {
            return;
        }
    }

    const TC x = (TC) src0_ptr[i0];
    T res;

    if (args.op == OP_UNARY_NUM_SCALE) {
        res = (T) (args.scale * x + args.bias);
    } else if (args.op == OP_UNARY_NUM_FILL) {
        res = (T) args.val;
    } else if (args.op == OP_UNARY_NUM_CLAMP) {
        res = (T) fmaxf((float)args.min, fminf((float)args.max, (float)x));
    } else if (args.op == OP_UNARY_NUM_SQR) {
        res = (T) (x * x);
    } else if (args.op == OP_UNARY_NUM_SQRT) {
        res = (T) sqrtf((float)x);
    } else if (args.op == OP_UNARY_NUM_SIN) {
        res = (T) sinf((float)x);
    } else if (args.op == OP_UNARY_NUM_COS) {
        res = (T) cosf((float)x);
    } else if (args.op == OP_UNARY_NUM_LOG) {
        res = (T) logf((float)x);
    } else if (args.op == OP_UNARY_NUM_LEAKY_RELU) {
        res = (T) (((float)x > 0.0f) ? (float)x : ((float)x * args.slope));
    } else if (args.op == OP_UNARY_NUM_TANH) {
        res = (T) tanhf((float)x);
    } else if (args.op == OP_UNARY_NUM_RELU) {
        res = (T) fmaxf(0.0f, (float)x);
    } else if (args.op == OP_UNARY_NUM_SIGMOID) {
        res = (T) (1.0f / (1.0f + expf(-(float)x)));
    } else if (args.op == OP_UNARY_NUM_GELU) {
        res = (T) (0.5f * (float)x * (1.0f + tanhf(SQRT_2_OVER_PI * (float)x * (1.0f + GELU_COEF_A * (float)x * (float)x))));
    } else if (args.op == OP_UNARY_NUM_GELU_ERF) {
        res = (T) (0.5f * (float)x * (1.0f + erf_approx(SQRT_2_INV * (float)x)));
    } else if (args.op == OP_UNARY_NUM_GELU_QUICK) {
        res = (T) ((float)x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * (float)x))));
    } else if (args.op == OP_UNARY_NUM_SILU) {
        res = (T) ((float)x / (1.0f + expf(-(float)x)));
    } else if (args.op == OP_UNARY_NUM_ELU) {
        res = (T) elu_approx((float)x);
    } else if (args.op == OP_UNARY_NUM_NEG) {
        res = (T) -(float)x;
    } else if (args.op == OP_UNARY_NUM_ABS) {
        res = (T) fabsf((float)x);
    } else if (args.op == OP_UNARY_NUM_SGN) {
        res = (T) (((float)x > 0.0f) ? 1.0f : (((float)x < 0.0f) ? -1.0f : 0.0f));
    } else if (args.op == OP_UNARY_NUM_STEP) {
        res = (T) (((float)x > 0.0f) ? 1.0f : 0.0f);
    } else if (args.op == OP_UNARY_NUM_HARDSWISH) {
        res = (T) ((float)x * fmaxf(0.0f, fminf(1.0f, (float)x / 6.0f + 0.5f)));
    } else if (args.op == OP_UNARY_NUM_HARDSIGMOID) {
        res = (T) fmaxf(0.0f, fminf(1.0f, (float)x / 6.0f + 0.5f));
    } else if (args.op == OP_UNARY_NUM_EXP) {
        res = (T) expf((float)x);
    } else if (args.op == OP_UNARY_NUM_SOFTPLUS) {
        res = (T) (((float)x > 20.0f) ? (float)x : logf(1.0f + expf((float)x)));
    } else if (args.op == OP_UNARY_NUM_EXPM1) {
        res = (T) (expf((float)x) - 1.0f);
    } else if (args.op == OP_UNARY_NUM_FLOOR) {
        res = (T) floorf((float)x);
    } else if (args.op == OP_UNARY_NUM_CEIL) {
        res = (T) ceilf((float)x);
    } else if (args.op == OP_UNARY_NUM_ROUND) {
        res = (T) roundf((float)x);
    } else if (args.op == OP_UNARY_NUM_TRUNC) {
        res = (T) truncf((float)x);
    } else if (args.op == OP_UNARY_NUM_XIELU) {
        const float xi      = (float)x;
        const float gate    = (xi > 0.0f) ? 1.0f : 0.0f;
        const float clamped = fminf(xi, args.val);
        const float y_pos   = args.scale * xi * xi + args.bias * xi;
        const float y_neg   = (expf(clamped) - 1.0f - xi) * args.slope + args.bias * xi;
        res = (T) (gate * y_pos + (1.0f - gate) * y_neg);
    }

    dst_ptr[i0] = res;
}

extern "C" __global__ void kernel_dsv4_softplus_sqrt_f32_4(
        ds4_hip_args_unary args,
        const char *src,
        char *dst) {
    const int k0 = blockIdx.x / args.ne01;
    const int i01 = blockIdx.x - k0 * args.ne01;
    const int i0 = k0 * blockDim.x + threadIdx.x;
    if (i0 >= args.ne0) return;

    const float4 *s = (const float4 *)(src + i01 * args.nb01);
          float4 *d = (      float4 *)(dst + i01 * args.nb1);
    const float4 x = s[i0];
    
    float4 sp;
    sp.x = (x.x > 20.0f) ? x.x : logf(1.0f + expf(x.x));
    sp.y = (x.y > 20.0f) ? x.y : logf(1.0f + expf(x.y));
    sp.z = (x.z > 20.0f) ? x.z : logf(1.0f + expf(x.z));
    sp.w = (x.w > 20.0f) ? x.w : logf(1.0f + expf(x.w));
    
    float4 res;
    res.x = sqrtf(sp.x);
    res.y = sqrtf(sp.y);
    res.z = sqrtf(sp.z);
    res.w = sqrtf(sp.w);
    
    d[i0] = res;
}

extern "C" __global__ void kernel_unary_f32_f32(
        ds4_hip_args_unary args,
        const char * src0,
        char * dst) {
    kernel_unary_impl<float, float, float>(args, src0, dst);
}

// Float4 overload for kernel_unary_impl
template <>
__device__ void kernel_unary_impl<float4, float4, float4>(
        ds4_hip_args_unary args,
        const char * src0,
        char * dst) {
    const float4 * src0_ptr;
          float4 * dst_ptr;

    int i0;

    if (args.cnt) {
        i0 = blockIdx.x;
        src0_ptr = (const float4 *) (src0);
        dst_ptr  = (      float4 *) (dst);
    } else {
        const int i03 = blockIdx.z;
        const int i02 = blockIdx.y;
        const int k0  = blockIdx.x / args.ne01;
        const int i01 = blockIdx.x - k0 * args.ne01;

        i0 = k0 * blockDim.x + threadIdx.x;

        src0_ptr = (const float4 *) (src0 + i03 * args.nb03 + i02 * args.nb02 + i01 * args.nb01);
        dst_ptr  = (      float4 *) (dst  + i03 * args.nb3  + i02 * args.nb2  + i01 * args.nb1 );
    }

    if (!args.cnt) {
        if (i0 >= args.ne0) {
            return;
        }
    }

    const float4 x = src0_ptr[i0];
    float4 res;

    // To keep it simple, process each component using a helper
    float vals[4] = {x.x, x.y, x.z, x.w};
    float r[4];

    for (int j = 0; j < 4; ++j) {
        float cx = vals[j];
        if (args.op == OP_UNARY_NUM_SCALE) {
            r[j] = args.scale * cx + args.bias;
        } else if (args.op == OP_UNARY_NUM_FILL) {
            r[j] = args.val;
        } else if (args.op == OP_UNARY_NUM_CLAMP) {
            r[j] = fmaxf(args.min, fminf(args.max, cx));
        } else if (args.op == OP_UNARY_NUM_SQR) {
            r[j] = cx * cx;
        } else if (args.op == OP_UNARY_NUM_SQRT) {
            r[j] = sqrtf(cx);
        } else if (args.op == OP_UNARY_NUM_SIN) {
            r[j] = sinf(cx);
        } else if (args.op == OP_UNARY_NUM_COS) {
            r[j] = cosf(cx);
        } else if (args.op == OP_UNARY_NUM_LOG) {
            r[j] = logf(cx);
        } else if (args.op == OP_UNARY_NUM_LEAKY_RELU) {
            r[j] = (cx > 0.0f) ? cx : (cx * args.slope);
        } else if (args.op == OP_UNARY_NUM_TANH) {
            r[j] = tanhf(cx);
        } else if (args.op == OP_UNARY_NUM_RELU) {
            r[j] = fmaxf(0.0f, cx);
        } else if (args.op == OP_UNARY_NUM_SIGMOID) {
            r[j] = 1.0f / (1.0f + expf(-cx));
        } else if (args.op == OP_UNARY_NUM_GELU) {
            r[j] = 0.5f * cx * (1.0f + tanhf(SQRT_2_OVER_PI * cx * (1.0f + GELU_COEF_A * cx * cx)));
        } else if (args.op == OP_UNARY_NUM_GELU_ERF) {
            r[j] = 0.5f * cx * (1.0f + erf_approx(SQRT_2_INV * cx));
        } else if (args.op == OP_UNARY_NUM_GELU_QUICK) {
            r[j] = cx * (1.0f / (1.0f + expf(GELU_QUICK_COEF * cx)));
        } else if (args.op == OP_UNARY_NUM_SILU) {
            r[j] = cx / (1.0f + expf(-cx));
        } else if (args.op == OP_UNARY_NUM_ELU) {
            r[j] = elu_approx(cx);
        } else if (args.op == OP_UNARY_NUM_NEG) {
            r[j] = -cx;
        } else if (args.op == OP_UNARY_NUM_ABS) {
            r[j] = fabsf(cx);
        } else if (args.op == OP_UNARY_NUM_SGN) {
            r[j] = (cx > 0.0f) ? 1.0f : ((cx < 0.0f) ? -1.0f : 0.0f);
        } else if (args.op == OP_UNARY_NUM_STEP) {
            r[j] = (cx > 0.0f) ? 1.0f : 0.0f;
        } else if (args.op == OP_UNARY_NUM_HARDSWISH) {
            r[j] = cx * fmaxf(0.0f, fminf(1.0f, cx / 6.0f + 0.5f));
        } else if (args.op == OP_UNARY_NUM_HARDSIGMOID) {
            r[j] = fmaxf(0.0f, fminf(1.0f, cx / 6.0f + 0.5f));
        } else if (args.op == OP_UNARY_NUM_EXP) {
            r[j] = expf(cx);
        } else if (args.op == OP_UNARY_NUM_SOFTPLUS) {
            r[j] = (cx > 20.0f) ? cx : logf(1.0f + expf(cx));
        } else if (args.op == OP_UNARY_NUM_EXPM1) {
            r[j] = expf(cx) - 1.0f;
        } else if (args.op == OP_UNARY_NUM_FLOOR) {
            r[j] = floorf(cx);
        } else if (args.op == OP_UNARY_NUM_CEIL) {
            r[j] = ceilf(cx);
        } else if (args.op == OP_UNARY_NUM_ROUND) {
            r[j] = roundf(cx);
        } else if (args.op == OP_UNARY_NUM_TRUNC) {
            r[j] = truncf(cx);
        } else if (args.op == OP_UNARY_NUM_XIELU) {
            const float gate    = (cx > 0.0f) ? 1.0f : 0.0f;
            const float clamped = fminf(cx, args.val);
            const float y_pos   = args.scale * cx * cx + args.bias * cx;
            const float y_neg   = (expf(clamped) - 1.0f - cx) * args.slope + args.bias * cx;
            r[j] = gate * y_pos + (1.0f - gate) * y_neg;
        }
    }
    
    res.x = r[0]; res.y = r[1]; res.z = r[2]; res.w = r[3];
    dst_ptr[i0] = res;
}

extern "C" __global__ void kernel_unary_f32_f32_4(
        ds4_hip_args_unary args,
        const char * src0,
        char * dst) {
    kernel_unary_impl<float4, float4, float4>(args, src0, dst);
}

extern "C" __global__ void kernel_unary_f16_f16(
        ds4_hip_args_unary args,
        const char * src0,
        char * dst) {
    kernel_unary_impl<half, half, float>(args, src0, dst);
}

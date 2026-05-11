#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../ds4_gpu_common.h"
#include <math.h>
#include <float.h>

#define WAVEFRONT_SIZE 32

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_xor(val, offset, WAVEFRONT_SIZE);
    return val;
}

typedef float4 float4_t;

__device__ inline float dot_f4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

extern "C" __global__ void kernel_dsv4_indexer_score_one_direct(
        const struct ds4_hip_args_dsv4_indexer_scores_fused args,
        const char *q,
        const char *weights,
        const char *index_comp,
        char *scores) {
    extern __shared__ char shared_mem[];
    uint row = blockIdx.x;
    ushort tid = threadIdx.x;
    ushort lane = threadIdx.x % WAVEFRONT_SIZE;
    ushort sg = threadIdx.x / WAVEFRONT_SIZE;

    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    float *ktg = (float*)shared_mem;        // [128]
    float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        const float *krow = (const float *)(index_comp + (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    __syncthreads();

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        const float4 *q4 = (const float4 *)(q + (uint64_t)head * args.q_head_stride);
        const float4 *k4 = (const float4 *)ktg;

        float s = dot_f4(q4[lane], k4[lane]);
        s = warp_reduce_sum(s);
        if (lane == 0) {
            const float *w = (const float *)weights;
            psum[sg] = fmaxf(s, 0.0f) * (w[head] * args.scale);
        }
        __syncthreads();
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float *dst = (float *)scores;
        dst[row] = acc;
    }
}

extern "C" __global__ void kernel_dsv4_router_weights_one(
        const char *probs,
        const char *selected,
        char *weights) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 6) return;

    const float *p = (const float *)probs;
    const int   *s = (const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = fmaxf(sum, 6.103515625e-5f);

    float *w = (float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

extern "C" __global__ void kernel_dsv4_router_finalize_one(
        const struct ds4_hip_args_dsv4_router_select_one args,
        const float *probs,
        const float *bias,
        const int32_t *hash,
        const int32_t *tokens,
        int32_t *selected) {
    extern __shared__ float scratch[];
    uint tid = threadIdx.x;

    if (tid >= 256) return;

    float *sel_scores = scratch;
    int32_t *idx = (int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    __syncthreads();

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                __syncthreads();
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
}

extern "C" __global__ void kernel_dsv4_topk_mask(
        const struct ds4_hip_args_dsv4_topk_mask args,
        const char * topk,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    *((float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

extern "C" __global__ void kernel_dsv4_topk_mask_scatter(
        const struct ds4_hip_args_dsv4_topk_mask args,
        const char * topk,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

extern "C" __global__ void kernel_dsv4_sort_i32_rows_asc(
        const struct ds4_hip_args_dsv4_topk_mask args,
        const char * src,
        char * dst) {
    extern __shared__ int32_t row_tmp[];
    uint row = blockIdx.x;
    uint tid = threadIdx.x;

    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= top_k) {
        return;
    }

    row_tmp[tid] = *((const int32_t *) (src + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01));
    __syncthreads();

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid && other < top_k) {
                const int32_t a = row_tmp[tid];
                const int32_t b = row_tmp[other];
                const bool up = (tid & k) == 0;
                if ((up && a > b) || (!up && a < b)) {
                    row_tmp[tid] = b;
                    row_tmp[other] = a;
                }
            }
            __syncthreads();
        }
    }

    *((int32_t *) (dst + (uint64_t)tid*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[tid];
}


struct float4_cast_half {
    float x, y, z, w;
    __device__ float4_cast_half(float4 v) {
        x = (float)(half)v.x;
        y = (float)(half)v.y;
        z = (float)(half)v.z;
        w = (float)(half)v.w;
    }
};

__device__ inline float dot_h4(float4_cast_half a, float4_cast_half b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

static __device__ inline void dsv4_attend_shared_f32_row_as_f16(
        const float4 *kv4,
        float4_cast_half q0,
        float4_cast_half q1,
        float4_cast_half q2,
        float4_cast_half q3,
        float scale,
        ushort lane,
        float &M,
        float &S,
        float4 &o0,
        float4 &o1,
        float4 &o2,
        float4 &o3) {
    const float4_cast_half k0 = float4_cast_half(kv4[lane +  0]);
    const float4_cast_half k1 = float4_cast_half(kv4[lane + 32]);
    const float4_cast_half k2 = float4_cast_half(kv4[lane + 64]);
    const float4_cast_half k3 = float4_cast_half(kv4[lane + 96]);

    float score = dot_h4(q0, k0) +
                  dot_h4(q1, k1) +
                  dot_h4(q2, k2) +
                  dot_h4(q3, k3);
    score = warp_reduce_sum(score) * scale;

    const float old_m = M;
    const float new_m = fmaxf(M, score);
    const float old_scale = expf(old_m - new_m);
    const float row_scale = expf(score - new_m);

    S = S * old_scale + row_scale;
    o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
    o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
    o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
    o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

    o0.x += k0.x * row_scale; o0.y += k0.y * row_scale; o0.z += k0.z * row_scale; o0.w += k0.w * row_scale;
    o1.x += k1.x * row_scale; o1.y += k1.y * row_scale; o1.z += k1.z * row_scale; o1.w += k1.w * row_scale;
    o2.x += k2.x * row_scale; o2.y += k2.y * row_scale; o2.z += k2.z * row_scale; o2.w += k2.w * row_scale;
    o3.x += k3.x * row_scale; o3.y += k3.y * row_scale; o3.z += k3.z * row_scale; o3.w += k3.w * row_scale;
    M = new_m;
}

static __device__ inline void dsv4_attend_shared_f32_row_as_f16_at(
        const float4 *kv4,
        uint row_in_tg,
        float4_cast_half q0,
        float4_cast_half q1,
        float4_cast_half q2,
        float4_cast_half q3,
        float scale,
        ushort lane,
        float &M,
        float &S,
        float4 &o0,
        float4 &o1,
        float4 &o2,
        float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static __device__ inline void dsv4_attend_sink(
        float score,
        float &M,
        float &S,
        float4 &o0,
        float4 &o1,
        float4 &o2,
        float4 &o3) {
    const float old_m = M;
    const float new_m = fmaxf(M, score);
    const float old_scale = expf(old_m - new_m);
    const float row_scale = expf(score - new_m);

    S = S * old_scale + row_scale;
    o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
    o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
    o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
    o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
    M = new_m;
}

extern "C" __global__ void kernel_dsv4_indexed_mixed_attention_heads8(
        const struct ds4_hip_args_dsv4_indexed_attention args,
        const char *q,
        const char *raw_kv,
        const char *comp_kv,
        const char *topk,
        const char *sinks,
        char *dst) {
    extern __shared__ float4 kv_shared_mem[];
    uint token = blockIdx.x;
    uint sg = threadIdx.x / WAVEFRONT_SIZE;
    uint head = blockIdx.y * 8u + sg;
    ushort tid = threadIdx.x;
    ushort lane = threadIdx.x % WAVEFRONT_SIZE;

    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    const float4 *q4 = (const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const float4_cast_half q0 = float4_cast_half(q4[lane +  0]);
    const float4_cast_half q1 = float4_cast_half(q4[lane + 32]);
    const float4_cast_half q2 = float4_cast_half(q4[lane + 64]);
    const float4_cast_half q3 = float4_cast_half(q4[lane + 96]);

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o2 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o3 = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            const float4 *src = (const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared_mem[tid] = src[tid];
            __syncthreads();
            dsv4_attend_shared_f32_row_as_f16(kv_shared_mem,
                                              q0, q1, q2, q3,
                                              args.scale,
                                              lane,
                                              M, S,
                                              o0, o1, o2, o3);
            __syncthreads();
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    const int32_t *row_topk = (const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        const float4 *src = (const float4 *)(comp_kv +
            (uint64_t)(uint)idx * args.comp_row_stride);
        if (tid < 128) kv_shared_mem[tid] = src[tid];
        __syncthreads();
        dsv4_attend_shared_f32_row_as_f16(kv_shared_mem,
                                          q0, q1, q2, q3,
                                          args.scale,
                                          lane,
                                          M, S,
                                          o0, o1, o2, o3);
        __syncthreads();
    }

    dsv4_attend_sink(((const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    float4 *dst4 = (float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    
    o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
    o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
    o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
    o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
    
    dst4[lane +  0] = o0;
    dst4[lane + 32] = o1;
    dst4[lane + 64] = o2;
    dst4[lane + 96] = o3;
}

extern "C" __global__ void kernel_dsv4_indexed_mixed_attention_heads8_rb4(
        const struct ds4_hip_args_dsv4_indexed_attention args,
        const char *q,
        const char *raw_kv,
        const char *comp_kv,
        const char *topk,
        const char *sinks,
        char *dst) {
    extern __shared__ float4 kv_shared_mem[];
    uint token = blockIdx.x;
    uint sg = threadIdx.x / WAVEFRONT_SIZE;
    uint head = blockIdx.y * 8u + sg;
    ushort tid = threadIdx.x;
    ushort lane = threadIdx.x % WAVEFRONT_SIZE;

    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    const float4 *q4 = (const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const float4_cast_half q0 = float4_cast_half(q4[lane +  0]);
    const float4_cast_half q1 = float4_cast_half(q4[lane + 32]);
    const float4_cast_half q2 = float4_cast_half(q4[lane + 64]);
    const float4_cast_half q3 = float4_cast_half(q4[lane + 96]);

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o2 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 o3 = {0.0f, 0.0f, 0.0f, 0.0f};

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 4u) {
            const uint n_rows = min(4u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += blockDim.x) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                const float4 *src = (const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared_mem[off] = src[c];
            }
            __syncthreads();
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_f32_row_as_f16_at(kv_shared_mem,
                                                     r,
                                                     q0, q1, q2, q3,
                                                     args.scale,
                                                     lane,
                                                     M, S,
                                                     o0, o1, o2, o3);
            }
            __syncthreads();
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    const int32_t *row_topk = (const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 4u) {
        uint rows[4];
        uint n_rows = 0;
        for (uint j = 0; j < 4u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += blockDim.x) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            const float4 *src = (const float4 *)(comp_kv +
                (uint64_t)rows[r] * args.comp_row_stride);
            kv_shared_mem[off] = src[c];
        }
        __syncthreads();
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_f32_row_as_f16_at(kv_shared_mem,
                                                 r,
                                                 q0, q1, q2, q3,
                                                 args.scale,
                                                 lane,
                                                 M, S,
                                                 o0, o1, o2, o3);
        }
        __syncthreads();
    }

    dsv4_attend_sink(((const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    float4 *dst4 = (float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
        
    o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
    o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
    o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
    o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
    
    dst4[lane +  0] = o0;
    dst4[lane + 32] = o1;
    dst4[lane + 64] = o2;
    dst4[lane + 96] = o3;
}

extern "C" __global__ void kernel_dsv4_indexer_scores_tiled_f32(
        const struct ds4_hip_args_dsv4_indexer_scores_fused args,
        const char *q,
        const char *weights,
        const char *index_comp,
        char *scores) {
    extern __shared__ char shared_mem[];
    uint c0 = blockIdx.x * 32u;
    uint t0 = blockIdx.y * 8u;
    ushort tid = threadIdx.x;
    ushort lane = threadIdx.x % WAVEFRONT_SIZE;
    ushort sg = threadIdx.x / WAVEFRONT_SIZE;

    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    float *qtg = (float*)shared_mem;             // [8][128]
    float *ktg = qtg + TM*D;         // [32][128]
    float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += blockDim.x) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                float *dst = (float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += blockDim.x) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            const float *row = (const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint r = lane / 4;
    const uint lc = (lane % 4) * 2;
    const uint col0 = sg * TS + lc;
    const uint col1 = sg * TS + lc + 1;
    
    const uint token0 = t0 + r;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    __syncthreads();

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += blockDim.x) {
            const uint q_r = i / D;
            const uint q_d = i - q_r*D;
            const uint token = t0 + q_r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                const float *qrow = (const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[q_d];
            }
            qtg[i] = v;
        }

        __syncthreads();

        float thread_dot0 = 0.0f;
        float thread_dot1 = 0.0f;
        for (uint db = 0; db < D/TS; db++) {
            #pragma unroll
            for (uint k = 0; k < TS; k++) {
                float q_val = qtg[r * D + db * TS + k];
                float k_val0 = ktg[col0 * D + db * TS + k];
                float k_val1 = ktg[col1 * D + db * TS + k];
                thread_dot0 += q_val * k_val0;
                thread_dot1 += q_val * k_val1;
            }
        }

        dot[r * TN + col0] = thread_dot0;
        dot[r * TN + col1] = thread_dot1;

        __syncthreads();

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            const float *w = (const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[r*TN + col0];
            acc0 += fmaxf(s, 0.0f) * (w[head] * args.scale);
        }
        if (token0 < args.n_tokens && comp1 < args.n_comp) {
            const float *w = (const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[r*TN + col1];
            acc1 += fmaxf(s, 0.0f) * (w[head] * args.scale);
        }

        __syncthreads();
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        float *dst = (float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token0 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        float *dst = (float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

extern "C" __global__ void kernel_dsv4_indexer_scores_tiled(
        const struct ds4_hip_args_dsv4_indexer_scores_fused args,
        const char *q,
        const char *weights,
        const char *index_comp,
        char *scores) {
    extern __shared__ char shared_mem[];
    uint c0 = blockIdx.x * 32u;
    uint t0 = blockIdx.y * 8u;
    ushort tid = threadIdx.x;
    ushort lane = threadIdx.x % WAVEFRONT_SIZE;
    ushort sg = threadIdx.x / WAVEFRONT_SIZE;

    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    half *qtg = (half *)shared_mem;             // [8][128]
    half *ktg = qtg + TM*D;                 // [32][128]
    float *dot = (float *)(ktg + TN*D);     // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += blockDim.x) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                float *dst = (float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += blockDim.x) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = __float2half(0.0f);
        if (comp < args.n_comp) {
            const float *row = (const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = __float2half(row[d]);
        }
        ktg[i] = v;
    }

    const uint r = lane / 4;
    const uint lc = (lane % 4) * 2;
    const uint col0 = sg * TS + lc;
    const uint col1 = sg * TS + lc + 1;
    
    const uint token0 = t0 + r;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    __syncthreads();

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += blockDim.x) {
            const uint q_r = i / D;
            const uint q_d = i - q_r*D;
            const uint token = t0 + q_r;
            half v = __float2half(0.0f);
            if (token < args.n_tokens) {
                const float *qrow = (const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = __float2half(qrow[q_d]);
            }
            qtg[i] = v;
        }

        __syncthreads();

        float thread_dot0 = 0.0f;
        float thread_dot1 = 0.0f;
        for (uint db = 0; db < D/TS; db++) {
            #pragma unroll
            for (uint k = 0; k < TS; k++) {
                float q_val = __half2float(qtg[r * D + db * TS + k]);
                float k_val0 = __half2float(ktg[col0 * D + db * TS + k]);
                float k_val1 = __half2float(ktg[col1 * D + db * TS + k]);
                thread_dot0 += q_val * k_val0;
                thread_dot1 += q_val * k_val1;
            }
        }

        dot[r * TN + col0] = thread_dot0;
        dot[r * TN + col1] = thread_dot1;

        __syncthreads();

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            const float *w = (const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[r*TN + col0];
            acc0 += fmaxf(s, 0.0f) * (w[head] * args.scale);
        }
        if (token0 < args.n_tokens && comp1 < args.n_comp) {
            const float *w = (const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[r*TN + col1];
            acc1 += fmaxf(s, 0.0f) * (w[head] * args.scale);
        }

        __syncthreads();
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        float *dst = (float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token0 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        float *dst = (float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

extern "C" __global__ void kernel_dsv4_indexer_weighted_sum(
        const struct ds4_hip_args_dsv4_indexer_weighted_sum args,
        const char * scores,
        const char * weights,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += fmaxf(s, 0.0f) * (w * args.scale);
    }

    *((float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

extern "C" __global__ void kernel_dsv4_softmax_pool(
        const struct ds4_hip_args_dsv4_softmax_pool args,
        const char * kv,
        const char * score,
        char * dst) {
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = fmaxf(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = expf(s - max_s);
        const float v = *((const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}

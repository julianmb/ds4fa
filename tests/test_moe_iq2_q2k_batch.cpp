#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <hip/hip_runtime.h>
#include "ds4_gpu.h"
#include "iq2_host_tables.inc"

#define QK_K 256
#define DS4_ROCM_MAX_N_EXPERT 256
#define DS4_ROCM_N_EXPERT_USED 6

typedef struct {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} test_block_iq2_xxs;

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
} test_block_q8_K;

static const uint8_t kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
static int8_t iq2xxs_signed_grid[256][128][8];

static void iq2xxs_signed_grid_init(void) {
    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(host_iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = host_ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

static inline float f16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } out;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7c00) >> 10;
    uint32_t mant = (h & 0x03ff);
    if (exp == 0) {
        if (mant == 0) {
            out.u = sign;
            return out.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        out.u = sign | 0x7f800000 | (mant << 13);
        return out.f;
    }
    exp = exp + (127 - 15);
    mant = mant << 13;
    out.u = sign | (exp << 23) | mant;
    return out.f;
}

static inline uint16_t f32_to_f16(float f) {
    union { float f; uint32_t u; } in;
    in.f = f;
    uint32_t sign = (in.u >> 16) & 0x8000;
    int32_t exp = ((in.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (in.u & 0x007fffff) >> 13;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static void quantize_row_q8_K(test_block_q8_K *y, const float *x, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float *xb = x + i * QK_K;
        float max_v = 0.0f;
        for (int j = 0; j < QK_K; j++) {
            float v = fabsf(xb[j]);
            if (v > max_v) max_v = v;
        }
        float d = max_v / 127.0f;
        y[i].d = d;
        float id = d > 0 ? 127.0f / max_v : 0.0f;
        for (int j = 0; j < QK_K; j++) {
            int q = (int)roundf(xb[j] * id);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            y[i].qs[j] = (int8_t)q;
        }
        for (int j = 0; j < QK_K / 16; j++) {
            int sum = 0;
            for (int l = 0; l < 16; l++) sum += y[i].qs[j * 16 + l];
            y[i].bsums[j] = (int16_t)sum;
        }
    }
}

static inline int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
}

static void cpu_vec_dot_iq2_xxs_q8_K(int n, float *s, const test_block_iq2_xxs *x, const test_block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint8_t *aux8 = (const uint8_t *)aux32;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    *s = 0.125f * sumf;
}

static float cpu_vec_dot_q2_K_q8_K(int n, const test_block_q2_K *x, const test_block_q8_K *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);
        int isum = 0;
        int is = 0;
        for (int k = 0; k < QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d0 = sc[is++] & 0x0f;
                for (int l = 0; l < 16; l++) {
                    int v0 = (q2[l] >> shift) & 3;
                    isum += d0 * v0 * (int)q8[l];
                }
                int d1 = sc[is++] & 0x0f;
                for (int l = 0; l < 16; l++) {
                    int v1 = (q2[16 + l] >> shift) & 3;
                    isum += d1 * v1 * (int)q8[16 + l];
                }
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

static float cpu_vec_dot_q2_K_f32(int n, const test_block_q2_K *x, const float *y) {
    const int nb = n / QK_K;
    float sumf = 0.0f;
    for (int b = 0; b < nb; b++) {
        const float d = f16_to_f32(x[b].d);
        const float dmin = f16_to_f32(x[b].dmin);
        for (int group = 0; group < 16; group++) {
            const uint8_t scale = x[b].scales[group];
            const float dl = d * (float)(scale & 0x0f);
            const float ml = dmin * (float)(scale >> 4);
            const int shift = 2 * ((group / 2) % 4);
            const uint8_t *q = x[b].qs + 32 * (group / 8) + 16 * (group & 1);
            const float *v = y + b * QK_K + group * 16;
            for (int i = 0; i < 16; i++) {
                sumf += (dl * (float)((q[i] >> shift) & 3) - ml) * v[i];
            }
        }
    }
    return sumf;
}

static int test_resident_iq2_prefill_matches_path_oracle(uint32_t n_tokens) {
    const uint32_t expert_in_dim = 2048;
    const uint32_t expert_mid_dim = 1024;
    const uint32_t out_dim = 2048;
    const uint32_t n_total_expert = 8;
    const uint32_t n_expert = 6;
    const float clamp = 0.0f;

    const uint64_t gate_row_bytes = (expert_in_dim / 256) * sizeof(test_block_iq2_xxs);
    const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
    const uint64_t down_row_bytes = (expert_mid_dim / 256) * sizeof(test_block_q2_K);
    const uint64_t down_expert_bytes = out_dim * down_row_bytes;

    uint64_t total_gate_bytes = n_total_expert * gate_expert_bytes;
    uint64_t total_down_bytes = n_total_expert * down_expert_bytes;

    uint8_t *h_gate = (uint8_t *)calloc(1, total_gate_bytes);
    uint8_t *h_up   = (uint8_t *)calloc(1, total_gate_bytes);
    uint8_t *h_down = (uint8_t *)calloc(1, total_down_bytes);

    srand(42);
    for (size_t i = 0; i < total_gate_bytes; i++) h_gate[i] = (uint8_t)(rand() & 0xff);
    for (size_t i = 0; i < total_gate_bytes; i++) h_up[i] = (uint8_t)(rand() & 0xff);
    for (size_t i = 0; i < total_down_bytes; i++) h_down[i] = (uint8_t)(rand() & 0xff);

    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (uint32_t r = 0; r < expert_mid_dim; r++) {
            test_block_iq2_xxs *gb = (test_block_iq2_xxs *)(h_gate + e * gate_expert_bytes + r * gate_row_bytes);
            test_block_iq2_xxs *ub = (test_block_iq2_xxs *)(h_up + e * gate_expert_bytes + r * gate_row_bytes);
            for (uint32_t b = 0; b < expert_in_dim / 256; b++) {
                gb[b].d = f32_to_f16(0.01f);
                ub[b].d = f32_to_f16(0.01f);
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            test_block_q2_K *db = (test_block_q2_K *)(h_down + e * down_expert_bytes + r * down_row_bytes);
            for (uint32_t b = 0; b < expert_mid_dim / 256; b++) {
                db[b].d = f32_to_f16(0.01f);
                db[b].dmin = f32_to_f16(0.005f);
            }
        }
    }

    float *h_x = (float *)calloc(n_tokens * expert_in_dim, sizeof(float));
    for (size_t i = 0; i < n_tokens * expert_in_dim; i++) h_x[i] = ((float)(rand() % 200) - 100.0f) / 100.0f;

    int32_t *h_selected = (int32_t *)calloc(n_tokens * n_expert, sizeof(int32_t));
    float *h_weights = (float *)calloc(n_tokens * n_expert, sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < n_expert; s++) {
            h_selected[t * n_expert + s] = s % n_total_expert;
            h_weights[t * n_expert + s] = 1.0f / n_expert;
        }
    }

    // CPU Reference Calculation
    float *cpu_out_q8 = (float *)calloc(n_tokens * out_dim, sizeof(float));
    float *cpu_out_f32 = (float *)calloc(n_tokens * out_dim, sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        test_block_q8_K *xq = (test_block_q8_K *)calloc(expert_in_dim / 256, sizeof(test_block_q8_K));
        quantize_row_q8_K(xq, h_x + t * expert_in_dim, expert_in_dim);

        for (uint32_t slot = 0; slot < n_expert; slot++) {
            uint32_t exp = h_selected[t * n_expert + slot];
            float w = h_weights[t * n_expert + slot];
            float *cpu_mid = (float *)calloc(expert_mid_dim, sizeof(float));

            for (uint32_t r = 0; r < expert_mid_dim; r++) {
                const test_block_iq2_xxs *gr = (const test_block_iq2_xxs *)(h_gate + exp * gate_expert_bytes + r * gate_row_bytes);
                const test_block_iq2_xxs *ur = (const test_block_iq2_xxs *)(h_up + exp * gate_expert_bytes + r * gate_row_bytes);
                float g_val = 0.0f, u_val = 0.0f;
                cpu_vec_dot_iq2_xxs_q8_K(expert_in_dim, &g_val, gr, xq);
                cpu_vec_dot_iq2_xxs_q8_K(expert_in_dim, &u_val, ur, xq);
                float silu = g_val / (1.0f + expf(-g_val));
                cpu_mid[r] = silu * u_val * w;
            }

            test_block_q8_K *midq = (test_block_q8_K *)calloc(expert_mid_dim / 256, sizeof(test_block_q8_K));
            quantize_row_q8_K(midq, cpu_mid, expert_mid_dim);
            for (uint32_t r = 0; r < out_dim; r++) {
                const test_block_q2_K *dr = (const test_block_q2_K *)(h_down + exp * down_expert_bytes + r * down_row_bytes);
                cpu_out_q8[t * out_dim + r] += cpu_vec_dot_q2_K_q8_K(expert_mid_dim, dr, midq);
                cpu_out_f32[t * out_dim + r] += cpu_vec_dot_q2_K_f32(expert_mid_dim, dr, cpu_mid);
            }
            free(cpu_mid);
            free(midq);
        }
        free(xq);
    }

    // GPU Calculation
    ds4_gpu_tensor *g_x = ds4_gpu_tensor_alloc(n_tokens * expert_in_dim * sizeof(float));
    ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc(n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *g_gate = ds4_gpu_tensor_alloc(n_tokens * n_expert * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *g_up = ds4_gpu_tensor_alloc(n_tokens * n_expert * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *g_mid = ds4_gpu_tensor_alloc(n_tokens * n_expert * expert_mid_dim * sizeof(float));
    ds4_gpu_tensor *g_down = ds4_gpu_tensor_alloc(n_tokens * n_expert * out_dim * sizeof(float));
    ds4_gpu_tensor *g_selected = ds4_gpu_tensor_alloc(n_tokens * n_expert * sizeof(int32_t));
    ds4_gpu_tensor *g_weights = ds4_gpu_tensor_alloc(n_tokens * n_expert * sizeof(float));

    ds4_gpu_tensor_write(g_x, 0, h_x, n_tokens * expert_in_dim * sizeof(float));
    ds4_gpu_tensor_write(g_selected, 0, h_selected, n_tokens * n_expert * sizeof(int32_t));
    ds4_gpu_tensor_write(g_weights, 0, h_weights, n_tokens * n_expert * sizeof(float));

    uint64_t model_size = total_gate_bytes * 2 + total_down_bytes;
    uint8_t *model_map = (uint8_t *)malloc(model_size);
    uint64_t gate_offset = 0;
    uint64_t up_offset = total_gate_bytes;
    uint64_t down_offset = total_gate_bytes * 2;
    memcpy(model_map + gate_offset, h_gate, total_gate_bytes);
    memcpy(model_map + up_offset, h_up, total_gate_bytes);
    memcpy(model_map + down_offset, h_down, total_down_bytes);

    bool mid_is_f16 = false;
    int ok = ds4_gpu_routed_moe_batch_tensor(
        g_out, g_gate, g_up, g_mid, g_down,
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        16 /* DS4_TENSOR_IQ2_XXS */, 10 /* DS4_TENSOR_Q2_K */,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        expert_in_dim, expert_mid_dim, out_dim,
        g_selected, g_weights,
        n_total_expert, n_expert,
        clamp, g_x, 0, n_tokens, &mid_is_f16, false);

    if (!ok) {
        fprintf(stderr, "ds4_gpu_routed_moe_batch_tensor failed!\n");
        return 0;
    }

    (void)hipDeviceSynchronize();

    float *gpu_out = (float *)calloc(n_tokens * out_dim, sizeof(float));
    ds4_gpu_tensor_read(g_out, 0, gpu_out, n_tokens * out_dim * sizeof(float));

    int nans = 0, q8_mismatches = 0, f32_mismatches = 0;
    double q8_sqerr = 0.0, f32_sqerr = 0.0;
    float q8_max_abs = 0.0f, f32_max_abs = 0.0f;
    for (uint32_t i = 0; i < n_tokens * out_dim; i++) {
        if (isnan(gpu_out[i]) || isinf(gpu_out[i])) {
            nans++;
        } else {
            const float q8_abs = fabsf(gpu_out[i] - cpu_out_q8[i]);
            const float f32_abs = fabsf(gpu_out[i] - cpu_out_f32[i]);
            q8_sqerr += (double)q8_abs * q8_abs;
            f32_sqerr += (double)f32_abs * f32_abs;
            if (q8_abs > q8_max_abs) q8_max_abs = q8_abs;
            if (f32_abs > f32_max_abs) f32_max_abs = f32_abs;
            if (q8_abs > 1.0f) q8_mismatches++;
            if (f32_abs > 1.0f) {
                f32_mismatches++;
                if (f32_mismatches <= 5) {
                    printf("  [n_tokens=%u] Float diff at %u: cpu=%f, gpu=%f\n",
                           n_tokens, i, cpu_out_f32[i], gpu_out[i]);
                }
            }
        }
    }

    const double count = (double)n_tokens * out_dim;
    printf("Results for n_tokens=%u: NaNs=%d, "
           "Q8 mismatches=%d rmse=%.6f max_abs=%.6f, "
           "F32 mismatches=%d rmse=%.6f max_abs=%.6f / %u\n",
           n_tokens, nans,
           q8_mismatches, sqrt(q8_sqerr / count), q8_max_abs,
           f32_mismatches, sqrt(f32_sqerr / count), f32_max_abs,
           n_tokens * out_dim);

    const float max_allowed = n_tokens == 1u ? 0.01f : 0.5f;
    const float path_max_abs = n_tokens == 1u ? q8_max_abs : f32_max_abs;
    const int passed = nans == 0 && path_max_abs <= max_allowed;
    printf("Path oracle for n_tokens=%u: %s (max_abs=%.6f, limit=%.6f)\n",
           n_tokens, passed ? "PASS" : "FAIL", path_max_abs, max_allowed);

    ds4_gpu_tensor_free(g_x);
    ds4_gpu_tensor_free(g_out);
    ds4_gpu_tensor_free(g_gate);
    ds4_gpu_tensor_free(g_up);
    ds4_gpu_tensor_free(g_mid);
    ds4_gpu_tensor_free(g_down);
    ds4_gpu_tensor_free(g_selected);
    ds4_gpu_tensor_free(g_weights);
    free(h_gate); free(h_up); free(h_down); free(h_x); free(h_selected); free(h_weights);
    free(cpu_out_q8); free(cpu_out_f32); free(gpu_out); free(model_map);
    return passed;
}

int main() {
    iq2xxs_signed_grid_init();
    printf("Initializing GPU backend...\n");
    if (!ds4_gpu_init()) {
        fprintf(stderr, "Failed to init GPU\n");
        return 1;
    }

    int failures = 0;
    failures += !test_resident_iq2_prefill_matches_path_oracle(1);
    failures += !test_resident_iq2_prefill_matches_path_oracle(2);
    failures += !test_resident_iq2_prefill_matches_path_oracle(4);
    failures += !test_resident_iq2_prefill_matches_path_oracle(16);

    ds4_gpu_cleanup();
    return failures == 0 ? 0 : 1;
}

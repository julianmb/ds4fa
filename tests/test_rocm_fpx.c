#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
static void *models[8];
static unsigned n_models;

static unsigned char *model_alloc(size_t bytes) {
    void *p = NULL;
    size_t rounded = (bytes + 4095) & ~(size_t)4095;
    CHECK(posix_memalign(&p, 4096, rounded) == 0);
    memset(p, 0, rounded);
    models[n_models++] = p;
    return p;
}

static ds4_gpu_tensor *upload(const void *p, size_t n) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(n);
    CHECK(t && ds4_gpu_tensor_write(t, 0, p, n));
    return t;
}

static void compare(const char *name, const float *a, const float *b, size_t n) {
    float worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == b[i]) continue; /* Includes untouched -INFINITY state rows. */
        float err = fabsf(a[i] - b[i]);
        CHECK(isfinite(a[i]) && isfinite(b[i]));
        if (err > worst) worst = err;
        if (err > 2e-4f * (1 + fabsf(b[i]))) {
            fprintf(stderr, "%s[%zu]: GPU %.9g reference %.9g\n", name, i, a[i], b[i]);
            exit(1);
        }
    }
    printf("PASS %s (%zu values, max error %.3g)\n", name, n, worst);
}

/* FP4 codes/scale are frozen known values, independently decoded into an
 * F32 APE table. Compare prefill and decode state with the existing F32 path. */
static void test_compressor(unsigned ratio) {
    const unsigned head = 32, width = ratio == 4 ? 64 : 32;
    const unsigned rows = ratio == 4 ? 8 : ratio, nt = ratio * 2 + 3;
    const size_t ape_bytes = ratio * width / 32 * 17;
    const size_t float_offset = (ape_bytes + 255) & ~(size_t)255;
    const size_t norm_offset = float_offset + ratio * width * sizeof(float);
    const size_t model_bytes = norm_offset + head * sizeof(float);
    unsigned char *model = model_alloc(model_bytes);
    float *ape = (float *)(model + float_offset);
    float *norm = (float *)(model + norm_offset);
    const float codebook[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    for (unsigned b = 0; b < ratio * width / 32; b++) {
        unsigned char *p = model + b * 17;
        p[16] = 64; /* UE4M3 half-scale exactly 1. */
        for (unsigned j = 0; j < 16; j++) {
            unsigned lo = (j + b) & 15, hi = (j * 3 + b + 1) & 15;
            p[j] = lo | (hi << 4);
            ape[b * 32 + j] = codebook[lo];
            ape[b * 32 + j + 16] = codebook[hi];
        }
    }
    for (unsigned j = 0; j < head; j++) norm[j] = 0.7f + j * 0.01f;
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    CHECK(ds4_gpu_cache_model_range(model, model_bytes, 0, model_bytes, "test model"));
    float *kv = malloc(nt * width * sizeof(float)), *sc = malloc(nt * width * sizeof(float));
    for (unsigned i = 0; i < nt * width; i++) {
        kv[i] = sinf(i * 0.13f);
        sc[i] = cosf(i * 0.17f);
    }
    ds4_gpu_tensor *k = upload(kv, nt * width * sizeof(float));
    ds4_gpu_tensor *s = upload(sc, nt * width * sizeof(float));
    const size_t comp_n = 3 * head, state_n = rows * width;
    ds4_gpu_tensor *comp[2], *state_k[2], *state_s[2];
    for (unsigned pass = 0; pass < 2; pass++) {
        comp[pass] = ds4_gpu_tensor_alloc(comp_n * sizeof(float));
        state_k[pass] = ds4_gpu_tensor_alloc(state_n * sizeof(float));
        state_s[pass] = ds4_gpu_tensor_alloc(state_n * sizeof(float));
        CHECK(comp[pass] && state_k[pass] && state_s[pass]);
        CHECK(ds4_gpu_tensor_fill_f32(comp[pass], 0, comp_n));
        CHECK(ds4_gpu_compressor_prefill_tensor(comp[pass], state_k[pass], state_s[pass],
            k, s, model, model_bytes, pass ? float_offset : 0, pass ? 0 : 101,
            norm_offset, 0, head, ratio, 0, nt, 0, 0, false,
            10000, 1, 0, 1, 32, 1, 1e-6f));
    }
    float *got = malloc(state_n * sizeof(float)), *ref = malloc(state_n * sizeof(float));
    ds4_gpu_tensor *a[3] = {comp[0], state_k[0], state_s[0]};
    ds4_gpu_tensor *b[3] = {comp[1], state_k[1], state_s[1]};
    for (unsigned i = 0; i < 3; i++) {
        size_t n = i ? state_n : comp_n;
        CHECK(ds4_gpu_tensor_read(a[i], 0, got, n * sizeof(float)));
        CHECK(ds4_gpu_tensor_read(b[i], 0, ref, n * sizeof(float)));
        char name[80]; snprintf(name, sizeof(name), "FP4 compressor ratio=%u field=%u", ratio, i);
        compare(name, got, ref, n);
    }
    for (unsigned i = 0; i < 2; i++) {
        ds4_gpu_tensor_free(comp[i]); ds4_gpu_tensor_free(state_k[i]); ds4_gpu_tensor_free(state_s[i]);
    }
    ds4_gpu_tensor_free(k); ds4_gpu_tensor_free(s);
    free(got); free(ref); free(kv); free(sc);
}

/* Dense projections include HC, compressor and indexer shapes. Keep the
 * packed matrix at the end of the model so an accidental F16 read is rejected. */
static void test_dense(unsigned in_dim) {
    const unsigned IN = in_dim;
    enum { OUT = 24, NT = 17 };
    const size_t row_bytes = IN / 32 * 17, bytes = OUT * row_bytes;
    unsigned char *model = model_alloc(bytes);
    float x[NT * IN], ref[NT * OUT], got[NT * OUT];
    const float cb[16] = {0,1,2,3,4,6,8,10,0,-1,-2,-3,-4,-6,-8,-10};
    for (size_t b = 0; b < bytes / 17; b++) {
        unsigned char *p = model + b * 17;
        for (unsigned j = 0; j < 16; j++) p[j] = (b * 29 + j * 19) & 255;
        p[16] = 48 + b % 24;
    }
    for (unsigned i = 0; i < NT * IN; i++) x[i] = sinf(i * 0.37f) * 0.03f;
    for (unsigned t = 0; t < NT; t++) for (unsigned r = 0; r < OUT; r++) {
        float sum = 0;
        for (unsigned i = 0; i < IN; i++) {
            const unsigned char *p = model + r * row_bytes + (i / 32) * 17;
            unsigned j = i % 32;
            unsigned code = (p[j % 16] >> (j < 16 ? 0 : 4)) & 15;
            float scale = (8 + (p[16] & 7)) * ldexpf(1.0f, (p[16] >> 3) - 11);
            sum += cb[code] * scale * x[t * IN + i];
        }
        ref[t * OUT + r] = sum;
    }
    CHECK(ds4_gpu_set_model_map(model, bytes));
    CHECK(ds4_gpu_cache_model_range(model, bytes, 0, bytes, "test dense"));
    ds4_gpu_tensor *xt = upload(x, sizeof(x));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(got));
    CHECK(out);
    CHECK(ds4_gpu_matmul_quant_tensor(out, model, bytes, 0, 101, IN, OUT, xt, NT));
    CHECK(ds4_gpu_tensor_read(out, 0, got, sizeof(got)));
    char name[80];
    snprintf(name, sizeof(name), "FP4 dense projection tokens=17 input=%u", IN);
    compare(name, got, ref, NT * OUT);
    ds4_gpu_tensor_free(xt); ds4_gpu_tensor_free(out);
}

/* Frozen reference tables and full 24-bit groups catch both byte crossings
 * in FP3. Nonconsecutive selected IDs catch expert addressing mistakes. */
static void test_moe(unsigned nt, unsigned dim) {
    const unsigned IN = dim, MID = dim;
    enum { OUT = 32, TOTAL = 3, USED = 2 };
    const size_t gr = IN / 32 * 10, dr = MID / 32 * 14;
    const size_t ge = MID * gr, de = OUT * dr;
    const size_t uoff = ge * TOTAL, doff = uoff * 2, bytes = doff + de * TOTAL;
    unsigned char *model = model_alloc(bytes);
    for (unsigned mat = 0; mat < 2; mat++) {
        for (size_t b = 0; b < TOTAL * ge / 10; b++) {
            unsigned char *p = model + mat * uoff + b * 10;
            for (unsigned j = 0; j < 8; j++) p[j] = (b * 37 + j * 13 + mat * 79) & 255;
            p[8] = 48 + b % 24; p[9] = (48 + (b * 3) % 24) | (b % 2 ? 128 : 0);
        }
    }
    for (size_t b = 0; b < TOTAL * de / 14; b++) {
        unsigned char *p = model + doff + b * 14;
        for (unsigned j = 0; j < 12; j++) p[j] = (b * 59 + j * 43 + 181) & 255;
        p[12] = p[13] = 64;
    }
    CHECK(ds4_gpu_set_model_map(model, bytes));
    CHECK(ds4_gpu_cache_model_range(model, bytes, 0, bytes, "test model"));
    float *x = malloc(nt * IN * sizeof(float)), *ref = calloc(nt * OUT, sizeof(float));
    int32_t *sel = malloc(nt * USED * sizeof(int32_t));
    float *weights = malloc(nt * USED * sizeof(float));
    for (unsigned i = 0; i < nt * IN; i++) x[i] = sinf(i * 0.7f) * 0.01f;
    for (unsigned i = 0; i < nt * USED; i++) { sel[i] = (i + 2) % TOTAL; weights[i] = 0.2f + 0.03f * i; }
    const int fp2[4] = {-1,0,1,2}, fp3[8] = {0,1,2,4,0,-1,-2,-4};
    for (unsigned t = 0; t < nt; t++) for (unsigned e = 0; e < USED; e++) {
        unsigned id = sel[t * USED + e]; float mid[MID];
        for (unsigned r = 0; r < MID; r++) {
            float gu[2] = {0,0};
            for (unsigned mat = 0; mat < 2; mat++) for (unsigned i = 0; i < IN; i++) {
                const unsigned char *p = model + mat * uoff + id * ge + r * gr + i / 32 * 10;
                unsigned code = (p[(i % 32) / 4] >> ((i % 4) * 2)) & 3;
                unsigned e = p[8 + (i % 32) / 16];
                float scale = (8 + (e & 7)) * ldexpf(1.0f, ((e & 127) >> 3) - 11);
                if (e & 128) scale = -scale;
                gu[mat] += fp2[code] * scale * x[t * IN + i];
            }
            if (gu[0] > 1) gu[0] = 1;
            gu[1] = fmaxf(-1, fminf(1, gu[1]));
            mid[r] = gu[0] / (1 + expf(-gu[0])) * gu[1] * weights[t * USED + e];
        }
        for (unsigned r = 0; r < OUT; r++) for (unsigned i = 0; i < MID; i++) {
            const unsigned char *p = model + doff + id * de + r * dr + i / 32 * 14 + (i % 32) / 8 * 3;
            unsigned packed = p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
            ref[t * OUT + r] += fp3[(packed >> ((i % 8) * 3)) & 7] * mid[i];
        }
    }
    ds4_gpu_tensor *xt = upload(x, nt * IN * sizeof(float));
    ds4_gpu_tensor *st = upload(sel, nt * USED * sizeof(int32_t));
    ds4_gpu_tensor *wt = upload(weights, nt * USED * sizeof(float));
    ds4_gpu_tensor *gt = ds4_gpu_tensor_alloc(nt * USED * MID * sizeof(float));
    ds4_gpu_tensor *ut = ds4_gpu_tensor_alloc(nt * USED * MID * sizeof(float));
    ds4_gpu_tensor *mt = ds4_gpu_tensor_alloc(nt * USED * MID * sizeof(float));
    ds4_gpu_tensor *dt = ds4_gpu_tensor_alloc(nt * USED * OUT * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(nt * OUT * sizeof(float));
    CHECK(gt && ut && mt && dt && out);
    bool mid_f16 = false;
    int ok = ds4_gpu_routed_moe_batch_tensor(out, gt, ut, mt, dt, model, bytes,
        0, uoff, doff, 107, 104, ge, gr, de, dr, IN, MID, OUT,
        st, wt, TOTAL, USED, 1, xt, 0, nt, &mid_f16, true);
    if (dim % 256 != 0) {
        /* The shared routed API requires dimensions divisible by 256. */
        CHECK(!ok);
        puts("PASS routed MoE rejects unsupported input=96");
    } else {
        CHECK(ok);
        float *got = malloc(nt * OUT * sizeof(float));
        CHECK(ds4_gpu_tensor_read(out, 0, got, nt * OUT * sizeof(float)));
        char name[80]; snprintf(name, sizeof(name), "FP2/FP3 routed MoE tokens=%u input=%u", nt, IN);
        compare(name, got, ref, nt * OUT);
        free(got);
    }
    ds4_gpu_tensor *all[] = {xt, st, wt, gt, ut, mt, dt, out};
    for (unsigned i = 0; i < sizeof(all)/sizeof(all[0]); i++) ds4_gpu_tensor_free(all[i]);
    free(x); free(sel); free(weights); free(ref);
}

int main(void) {
    CHECK(ds4_gpu_init());
    test_dense(256); test_dense(96);
    test_compressor(4); test_compressor(128);
    test_moe(1, 256); test_moe(3, 256); test_moe(1, 96);
    ds4_gpu_cleanup();
    for (unsigned i = 0; i < n_models; i++) free(models[i]);
    puts("ROCmFPX regression PASSED");
    return 0;
}

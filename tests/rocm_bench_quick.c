/*
 * Quick ROCm kernel-execution + bandwidth bench for Strix Halo.
 *
 * Unlike the allocation-only smoke test, this actually launches compute kernels
 * (tensor fill + copy) on gfx1151 and reports achievable bandwidth, so a bad
 * ISA / missing kernel path fails loudly instead of silently.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ds4_gpu.h"

#define GiB (1073741824.0)

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "rocm-bench-quick: FAIL: %s\n", (msg)); \
        g_failed = 1; \
        goto cleanup; \
    } \
} while (0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void) {
    CHECK(ds4_gpu_init(), "ds4_gpu_init");

    const uint64_t elems = 64 * 1024 * 1024; /* 256 MiB of f32 */
    const uint64_t bytes = elems * sizeof(float);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
    CHECK(a, "alloc a");
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
    CHECK(b, "alloc b");

    /* Warm up + correctness: fill, copy, read back, verify. */
    CHECK(ds4_gpu_tensor_fill_f32(a, 2.0f, elems), "fill a");
    CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, bytes), "copy a->b");
    CHECK(ds4_gpu_synchronize(), "sync");
    float sample = 0;
    CHECK(ds4_gpu_tensor_read(b, 0, &sample, sizeof(sample)), "read b");
    CHECK(sample == 2.0f, "copy correctness (expected 2.0)");

    /* Timed pass: repeated fill+copy to measure device bandwidth. */
    const int iters = 20;
    const double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        CHECK(ds4_gpu_tensor_fill_f32(a, (float)(i + 1), elems), "fill");
        CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, bytes), "copy");
    }
    CHECK(ds4_gpu_synchronize(), "sync timed");
    const double t1 = now_sec();
    const double secs = t1 - t0;
    /* Each iter moves 2*bytes (fill writes a, copy reads a writes b). */
    const double gib_s = (double)iters * 2.0 * (double)bytes / GiB / secs;
    fprintf(stderr, "rocm-bench-quick: %.1f GiB in %.3f s => %.1f GiB/s "
            "(fill+copy, gfx1151)\n",
            (double)iters * 2.0 * (double)bytes / GiB, secs, gib_s);

    /* Sanity floor: if kernels don't really run we'd see implausibly low or
     * garbage; require a minimal throughput to catch a stalled device. */
    if (gib_s < 1.0) {
        fprintf(stderr, "rocm-bench-quick: FAIL: throughput implausibly low\n");
        g_failed = 1;
    }

cleanup:
    ds4_gpu_cleanup();
    if (g_failed) {
        fprintf(stderr, "rocm-bench-quick: FAILED\n");
        return 1;
    }
    fprintf(stderr, "rocm-bench-quick: PASSED\n");
    return 0;
}

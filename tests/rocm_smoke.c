/*
 * ROCm / Strix Halo smoke test for the current ds4_gpu API.
 *
 * Exercises the GPU backend allocation and copy paths plus the model-range
 * mapping helpers without requiring any model weights. Every error path cleans
 * up GPU state, and the page-aligned host model range is deliberately freed
 * only after ds4_gpu_cleanup() so the unregister ordering is exercised.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ds4_gpu.h"

#define STRIXHALO_GIB (1073741824ull)

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "rocm-smoke: FAIL: %s\n", (msg)); \
        g_failed = 1; \
        goto cleanup; \
    } \
} while (0)

static void *alloc_page_aligned(uint64_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 4096, (size_t)bytes) != 0) return NULL;
    return p;
}

/* Map a real GGUF file for the optional real-model smoke path. Returns the base
 * pointer and sets *size, or NULL if the file is absent/unreadable. */
static void *map_model_file(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    if (st.st_size == 0) { close(fd); return NULL; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *size = (uint64_t)st.st_size;
    return p;
}

int main(void) {
    void *host_model = NULL;
    int ok = ds4_gpu_init();
    CHECK(ok, "ds4_gpu_init");

    /* Device tensor: allocate, write, fill, copy, synchronize, read. */
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(1024 * sizeof(float));
    CHECK(a, "tensor alloc");
    CHECK(ds4_gpu_tensor_fill_f32(a, 1.0f, 1024), "tensor fill");

    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(1024 * sizeof(float));
    CHECK(b, "tensor alloc b");

    float host_src[1024];
    for (int i = 0; i < 1024; i++) host_src[i] = (float)(i + 1);
    CHECK(ds4_gpu_tensor_write(a, 0, host_src, sizeof(host_src)), "tensor write");

    CHECK(ds4_gpu_tensor_copy(b, 0, a, 0, sizeof(host_src)), "tensor copy");

    CHECK(ds4_gpu_synchronize(), "synchronize");

    float host_dst[1024];
    memset(host_dst, 0, sizeof(host_dst));
    CHECK(ds4_gpu_tensor_read(b, 0, host_dst, sizeof(host_dst)), "tensor read");
    for (int i = 0; i < 1024; i++) {
        CHECK(host_dst[i] == (float)(i + 1), "tensor copy round-trip mismatch");
    }

    /* Managed tensor allocation. */
    ds4_gpu_tensor *m = ds4_gpu_tensor_alloc_managed(256 * sizeof(float));
    CHECK(m, "managed tensor alloc");
    ds4_gpu_tensor_free(m);
    m = NULL;

    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(b);
    a = b = NULL;

    /* Page-aligned host model range + mapping/cache helpers. */
    const uint64_t model_bytes = 16 * 1024 * 1024ull;
    host_model = alloc_page_aligned(model_bytes);
    CHECK(host_model, "page-aligned host model alloc");
    memset(host_model, 0, (size_t)model_bytes);

    CHECK(ds4_gpu_set_model_map(host_model, model_bytes), "set_model_map");
    CHECK(ds4_gpu_cache_model_range(host_model, model_bytes, 0, model_bytes, "smoke"),
          "cache_model_range");

    /* Optional real-model path: if DS4_TEST_MODEL points at a GGUF, map it and
     * cache a real range so the smoke test doubles as a "can I load a model"
     * gate. The mapping is freed after ds4_gpu_cleanup(). */
    const char *real_model = getenv("DS4_TEST_MODEL");
    void *real_base = NULL;
    uint64_t real_size = 0;
    if (real_model && real_model[0] != '\0') {
        real_base = map_model_file(real_model, &real_size);
        if (real_base) {
            fprintf(stderr, "rocm-smoke: mapping real model %s (%.1f MiB)\n",
                    real_model, (double)real_size / (1024.0 * 1024.0));
            CHECK(ds4_gpu_set_model_map(real_base, real_size), "set_model_map (real)");
            CHECK(ds4_gpu_cache_model_range(real_base, real_size, 0, real_size, "smoke-real"),
                  "cache_model_range (real)");
        } else {
            fprintf(stderr, "rocm-smoke: DS4_TEST_MODEL=%s not available, skipping real-model path\n",
                    real_model);
        }
    }

cleanup:
    /* Cleanup unregisters the host mapping; free the host range only after. */
    ds4_gpu_cleanup();
    if (host_model) free(host_model);
    if (real_base) munmap(real_base, (size_t)real_size);

    if (g_failed) {
        fprintf(stderr, "rocm-smoke: FAILED\n");
        return 1;
    }
    fprintf(stderr, "rocm-smoke: PASSED\n");
    return 0;
}

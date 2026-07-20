/*
 * rocm_model_fit.c: report whether DS4_TEST_MODEL fits in the Strix Halo
 * TTM/GTT mapping limit, using the engine's own per-model estimate.
 *
 * Exit 0 = fits (with headroom), 1 = would OOM / error, 2 = no DS4_TEST_MODEL.
 * Honors the same env vars as the rest of the tooling (DS4_ROCM_TTM_PAGES,
 * DS4_ROCM_TTM_AUTORAISE, DS4_ROCM_DIAG, ...). When autoraise is enabled and
 * succeeds, the verdict reflects the raised limit.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "ds4_gpu.h"

static void print_verdict(const ds4_rocm_model_load_estimate *e) {
    const double mb = 1024.0 * 1024.0;
    fprintf(stderr, "rocm-model-fit: model=%.1f MiB\n", (double)e->model_bytes / mb);
    fprintf(stderr, "rocm-model-fit: ttm_limit=%.1f MiB\n", (double)e->ttm_limit_bytes / mb);
    if (e->gguf_magic[0])
        fprintf(stderr, "rocm-model-fit: gguf=%s tensors=%llu\n",
                e->gguf_magic, (unsigned long long)e->gguf_tensor_count);
    if (e->would_have_oomed) {
        fprintf(stderr,
                "rocm-model-fit: VERDICT: DOES NOT FIT (need %.1f MiB, limit %.1f MiB)\n",
                (double)e->model_bytes / mb, (double)e->ttm_limit_bytes / mb);
        return;
    }
    fprintf(stderr, "rocm-model-fit: VERDICT: FITS (headroom %.1f MiB%s)\n",
            (double)e->headroom_bytes / mb,
            e->headroom_below_8g ? ", below 8 GiB reserve" : "");
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || model[0] == '\0') {
        fprintf(stderr, "rocm-model-fit: set DS4_TEST_MODEL=/path/to/model.gguf\n");
        return 2;
    }

    int fd = open(model, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "rocm-model-fit: cannot open %s\n", model);
        return 1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        fprintf(stderr, "rocm-model-fit: bad file %s\n", model);
        return 1;
    }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "rocm-model-fit: mmap failed for %s\n", model);
        return 1;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "rocm-model-fit: ds4_gpu_init failed\n");
        munmap(p, (size_t)st.st_size);
        return 1;
    }

    if (!ds4_gpu_set_model_map(p, (uint64_t)st.st_size)) {
        fprintf(stderr, "rocm-model-fit: set_model_map failed\n");
        ds4_gpu_cleanup();
        munmap(p, (size_t)st.st_size);
        return 1;
    }

    const ds4_rocm_model_load_estimate *e = ds4_rocm_last_model_load_estimate();
    print_verdict(e);

    const int fits = !e->would_have_oomed;
    ds4_gpu_cleanup();
    munmap(p, (size_t)st.st_size);
    return fits ? 0 : 1;
}

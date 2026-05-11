#include "ds4_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <mutex>

/* =========================================================================
 * AMD XDNA 2 NPU Implementation (XRT 2026.1)
 * ========================================================================= */

#if __has_include(<xrt/xrt_device.h>) && __has_include(<xrt/xrt_bo.h>)
#define DS4_HAS_XRT 1
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#else
#define DS4_HAS_XRT 0
#endif

struct ds4_npu_context {
#if DS4_HAS_XRT
    xrt::device device;
    xrt::uuid   uuid;
    xrt::kernel mtp_kernel;
    // Cache for zero-copy Buffer Objects indexed by host pointer
    std::map<void*, xrt::bo> bo_cache;
    std::mutex mtx;
#else
    int dummy;
#endif
};

extern "C" {

int ds4_npu_init(ds4_npu_context **ctx_out, const char *xclbin_path) {
    if (!ctx_out) return 0;
#if !DS4_HAS_XRT
    fprintf(stderr, "ds4_npu: XRT SDK not found during compilation. NPU disabled.\n");
    *ctx_out = NULL;
    return 0;
#else
    try {
        ds4_npu_context *ctx = new ds4_npu_context();
        
        // 1. Discover XDNA 2 device (typically 0)
        ctx->device = xrt::device(0);
        
        // 2. Load the compiled MTP graph
        if (xclbin_path && xclbin_path[0] != '\0') {
            ctx->uuid = ctx->device.load_xclbin(xclbin_path);
            // 3. Instantiate the speculative draft kernel
            // In a real DeepSeek MTP xclbin, the kernel is usually named "mtp_gen"
            ctx->mtp_kernel = xrt::kernel(ctx->device, ctx->uuid, "mtp_gen");
            printf("ds4_npu: Loaded %s into XDNA 2 NPU.\n", xclbin_path);
        } else {
            printf("ds4_npu: Initialized NPU device (no graph loaded yet).\n");
        }

        *ctx_out = ctx;
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "ds4_npu: Failed to initialize NPU: %s\n", e.what());
        return 0;
    }
#endif
}

int ds4_npu_register_buffer(ds4_npu_context *ctx, void *host_ptr, size_t size) {
    if (!ctx || !host_ptr) return 0;
#if DS4_HAS_XRT
    try {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        
        // Create an XRT buffer object that wraps the existing host pointer (Zero-Copy).
        // The host_ptr should have been allocated via hipHostMalloc for best results.
        // We use flags::normal which on XDNA 2 implies pinning the user pointer.
        xrt::bo buffer = xrt::bo(ctx->device, host_ptr, size, xrt::bo::flags::normal, 0);
        
        ctx->bo_cache[host_ptr] = buffer;
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "ds4_npu: Buffer registration failed: %s\n", e.what());
        return 0;
    }
#else
    return 0;
#endif
}

void ds4_npu_unregister_buffer(ds4_npu_context *ctx, void *host_ptr) {
    if (!ctx || !host_ptr) return;
#if DS4_HAS_XRT
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->bo_cache.erase(host_ptr);
#endif
}

int ds4_npu_draft_mtp(ds4_npu_context *ctx, const int *input_tokens, int num_input, int *output_tokens, int max_draft) {
    if (!ctx || !input_tokens || !output_tokens) return -1;
#if !DS4_HAS_XRT
    return -1;
#else
    try {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        
        // 1. Resolve BOs for input/output
        auto it_in = ctx->bo_cache.find((void*)input_tokens);
        auto it_out = ctx->bo_cache.find((void*)output_tokens);
        
        if (it_in == ctx->bo_cache.end() || it_out == ctx->bo_cache.end()) {
            return -1; // Buffers not registered
        }

        xrt::bo &bo_in = it_in->second;
        xrt::bo &bo_out = it_out->second;

        // 2. Sync to ensure NPU sees latest CPU/iGPU writes
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 3. Launch the MTP draft module asynchronously
        // In a typical NPU kernel, we pass input buffer, output buffer, 
        // and current context length.
        auto run = ctx->mtp_kernel(bo_in, bo_out, num_input, max_draft);
        
        // 4. Wait for NPU to complete its predictions
        run.wait();

        // 5. Sync back so CPU/iGPU can read drafted tokens
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // We assume the kernel writes the number of tokens successfully 
        // drafted into the first word of the output or returns it.
        // For this implementation, we'll return max_draft for now.
        return max_draft;
    } catch (const std::exception& e) {
        fprintf(stderr, "ds4_npu: Draft execution failed: %s\n", e.what());
        return -1;
    }
#endif
}
void ds4_npu_cleanup(ds4_npu_context *ctx) {
    if (ctx) {
#if DS4_HAS_XRT
        delete ctx;
#else
        free(ctx);
#endif
    }
}

int ds4_npu_status_check(void) {
#if !DS4_HAS_XRT
    return 0;
#else
    try {
        xrt::device d(0);
        printf("ds4_npu: AMD XDNA Device Detected: %s\n", d.get_info<xrt::info::device::name>().c_str());
        return 1;
    } catch (...) {
        return 0;
    }
#endif
}

} // extern "C"

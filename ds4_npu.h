#ifndef DS4_NPU_H
#define DS4_NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context for the NPU execution layer.
 * Manages the XRT device, memory bank contexts, and compiled graphs. */
typedef struct ds4_npu_context ds4_npu_context;

/* Initialize the NPU device and load the compiled MTP graph.
 * xclbin_path: Path to the compiled NPU binary (.xclbin).
 * Returns 1 on success, 0 on failure. */
int ds4_npu_init(ds4_npu_context **ctx_out, const char *xclbin_path);

/* Register a host pointer (allocated via hipHostMalloc) for zero-copy NPU access.
 * This utilizes AMD's User Pointer (userptr) pinning. */
int ds4_npu_register_buffer(ds4_npu_context *ctx, void *host_ptr, size_t size);

/* De-register a buffer and unpin it from the NPU DMA engine. */
void ds4_npu_unregister_buffer(ds4_npu_context *ctx, void *host_ptr);

/* Execute the native DeepSeek MTP draft module on the NPU asynchronously using Ring Buffers.
 * This utilizes a double-buffered zero-copy state to allow the NPU to draft token t+2 
 * while the iGPU is still verifying token t+1.
 * input_tokens: Current context tokens.
 * num_input: Length of context.
 * output_tokens: Buffer where drafted token IDs will be written.
 * max_draft: Max tokens to predict (e.g. 4-8).
 * buffer_idx: Which ring buffer to use (0 or 1).
 * Returns the number of drafted tokens, or -1 on error. */
int ds4_npu_draft_mtp_async(ds4_npu_context *ctx, const int *input_tokens, int num_input, int *output_tokens, int max_draft, int buffer_idx);

/* Cleanup and release NPU resources. */
void ds4_npu_cleanup(ds4_npu_context *ctx);

/* Check if NPU is available and print its name. */
int ds4_npu_status_check(void);

#ifdef __cplusplus
}
#endif

#endif

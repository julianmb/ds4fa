import re
with open('ds4fa/ds4_hip.cpp', 'r') as f:
    text = f.read()

replacement = """int ds4_hip_set_model_map(const void *model_map, uint64_t model_size) {
    return 1;
}

int ds4_hip_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    (void)model_size;
    
    // Sniff memory to see if we can safely pin and pre-warm this block
    size_t free_mem, total_mem;
    bool can_pin = false;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
        // Leave a 4GB buffer for OS and scratch
        if (map_size < (free_mem - (4ULL * 1024 * 1024 * 1024))) {
            can_pin = true;
        }
    }

    if (can_pin) {
        void *target_ptr = (void *)((const char *)model_map + map_offset);
        if (hipHostRegister(target_ptr, map_size, hipHostRegisterDefault) == hipSuccess) {
            // Hint: Read-mostly optimized for Strix Halo iGPU L2 / System Cache
            hipMemAdvise(target_ptr, map_size, hipMemAdviseSetReadMostly, 0);
            
            // GPU Pre-warming: Touch each page to force HW address translation and cache warm-up
            uint64_t num_pages = (map_size + 4095) / 4096;
            dim3 block(256);
            dim3 grid((num_pages + 255) / 256);
            kernel_prewarm<<<grid, block, 0, g_stream>>>((const char *)target_ptr, map_size);
            hipStreamSynchronize(g_stream);
            fprintf(stderr, "ds4_hip: Model mapped, pinned, and pre-warmed successfully.\\n");
        } else {
            fprintf(stderr, "ds4_hip: hipHostRegister failed, falling back to unpinned HMM memory.\\n");
        }
    } else {
        fprintf(stderr, "ds4_hip: Model size (%.1f GB) exceeds safe free memory. Skipping pinning to allow Pipeline Parallel Q4 execution via HMM.\\n", (double)map_size / (1024.0*1024.0*1024.0));
    }
    
    return 1;
}"""

# Find the old functions
text = re.sub(r'int ds4_hip_set_model_map\(const void \*model_map, uint64_t model_size\) \{.*?\n\}', '', text, flags=re.DOTALL)
text = re.sub(r'int ds4_hip_set_model_map_range\(const void \*model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size\) \{.*?\n\}', replacement, text, flags=re.DOTALL)


with open('ds4fa/ds4_hip.cpp', 'w') as f:
    f.write(text)

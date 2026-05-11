import re
with open('ds4fa/ds4_hip.cpp', 'r') as f:
    text = f.read()

guard_code = """    // Hardware Guard: Ensure Strix Halo limits are respected
    size_t free_mem, total_mem;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
        if (total_mem <= (size_t)140 * 1024 * 1024 * 1024) { // <= ~130GB usually means 128GB APU
            fprintf(stderr, "ds4_hip: 128GB Strix Halo APU detected. Defaulting to safe memory profiles (q2 support).\\n");
            // Capping the memory pool aggressively
            hipMemPool_t mem_pool;
            if (hipDeviceGetDefaultMemPool(&mem_pool, 0) == hipSuccess) {
                uint64_t threshold = (uint64_t)1 * 1024 * 1024 * 1024; // Restrict to 1GB to prevent OS starvation
                hipMemPoolSetAttribute(mem_pool, hipMemPoolAttrReleaseThreshold, &threshold);
            }
        } else {
            // Larger APUs or Pipeline Parallel configurations (e.g. 256GB configurations)
            fprintf(stderr, "ds4_hip: Large memory system detected (>128GB). Enabling q4 support profiles.\\n");
            hipMemPool_t mem_pool;
            if (hipDeviceGetDefaultMemPool(&mem_pool, 0) == hipSuccess) {
                uint64_t threshold = (uint64_t)2 * 1024 * 1024 * 1024; // 2GB
                hipMemPoolSetAttribute(mem_pool, hipMemPoolAttrReleaseThreshold, &threshold);
            }
        }
    }"""

text = re.sub(r'    // Hardware Guard: Ensure Strix Halo 128GB limits are respected by default.*?\}\n    \}', guard_code, text, flags=re.DOTALL)

with open('ds4fa/ds4_hip.cpp', 'w') as f:
    f.write(text)

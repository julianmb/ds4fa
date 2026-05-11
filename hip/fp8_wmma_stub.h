// Foundational struct for FP8 WMMA extension
typedef uint32_t wmma_fp8v4 __attribute__((vector_size(16)));

__device__ inline float8_to_f16(uint8_t val) {
    // Stub for FP8 E4M3 to F16 conversion if needed
    return 0.0f;
}

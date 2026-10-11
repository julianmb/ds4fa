#include <stdint.h>
#include <math.h>

/* Wave32 ROCmFP2/ROCmFP3 routed path. GGUF 107 uses 10-byte
 * blocks (8 packed code bytes, two signed UE4M3 half-scales); GGUF 104 uses 14-byte
 * blocks (12 packed 3-bit code bytes, two UE4M3 half-scales). */
__device__ __forceinline__ static float dev_rocmfpx_ue4m3(uint8_t e) {
    if (e > 0x7eu) return 0.0f;
    const uint32_t ex = e >> 3u, mant = e & 7u;
    if (ex == 0u) return (float)mant * 0x1p-10f;
    return (1.0f + (float)mant * 0.125f) * __uint_as_float((ex + 119u) << 23u);
}

__global__ static void moe_gate_up_mid_rocmfpx23_kernel(
        float *gate_out, float *up_out, float *mid_out,
        const char *gate_base, const char *up_base,
        const char * const *gate_slots, const char * const *up_slots,
        const float *x, const int32_t *selected, const float *weights,
        uint64_t expert_bytes, uint64_t row_bytes, uint32_t in_dim,
        uint32_t mid_dim, uint32_t n_expert, float clamp) {
    const uint32_t pair = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    if (row >= mid_dim) return;
    const uint32_t tok = pair / n_expert, slot = pair % n_expert;
    int32_t si = selected[(uint64_t)tok * n_expert + slot];
    if (si < 0) si = 0;
    const char *gb = gate_slots ? gate_slots[(uint32_t)si] : gate_base + (uint64_t)(uint32_t)si * expert_bytes;
    const char *ub = up_slots ? up_slots[(uint32_t)si] : up_base + (uint64_t)(uint32_t)si * expert_bytes;
    const uint8_t *gr = (const uint8_t *)gb + (uint64_t)row * row_bytes;
    const uint8_t *ur = (const uint8_t *)ub + (uint64_t)row * row_bytes;
    const float *xp = x + (uint64_t)tok * in_dim;
    float g = 0.0f, u = 0.0f;
    /* Four lanes per packed block, eight blocks per wave. Reuse one
     * signed half-scale for eight adjacent values per lane. */
    for (uint32_t base = 0; base < in_dim / 32u; base += 8u) {
        const uint32_t block = base + lane / 4u;
        if (block >= in_dim / 32u) continue;
        const uint8_t *gp = gr + (uint64_t)block * 10u;
        const uint8_t *up = ur + (uint64_t)block * 10u;
        const uint32_t j0 = (lane & 3u) * 8u;
        const uint32_t gc = (uint32_t)gp[j0 / 4u] | ((uint32_t)gp[j0 / 4u + 1u] << 8u);
        const uint32_t uc = (uint32_t)up[j0 / 4u] | ((uint32_t)up[j0 / 4u + 1u] << 8u);
        /* Original Lucebox type-107 codebook; S40/affine variants that
         * reuse this ID are incompatible with the supported artifact. */
        const uint8_t ge = gp[8u + j0 / 16u], ue = up[8u + j0 / 16u];
        const float gs = dev_rocmfpx_ue4m3(ge & 127u) * ((ge & 128u) ? -1.0f : 1.0f);
        const float us = dev_rocmfpx_ue4m3(ue & 127u) * ((ue & 128u) ? -1.0f : 1.0f);
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const int gv = (int)((gc >> (j * 2u)) & 3u) - 1;
            const int uv = (int)((uc >> (j * 2u)) & 3u) - 1;
            const float xv = xp[block * 32u + j0 + j];
            g += (float)gv * gs * xv;
            u += (float)uv * us * xv;
        }
    }
    g = warp_sum_f32(g);
    u = warp_sum_f32(u);
    if (lane != 0u) return;
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    const uint64_t off = (uint64_t)pair * mid_dim + row;
    gate_out[off] = g;
    up_out[off] = u;
    mid_out[off] = (g / (1.0f + expf(-g))) * u * weights[(uint64_t)tok * n_expert + slot];
}

__global__ static void moe_down_sum_rocmfpx3_kernel(
        float *out, const char *down_base, const char * const *down_slots,
        const float *mid, const int32_t *selected, uint64_t expert_bytes,
        uint64_t row_bytes, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_expert) {
    const uint32_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x / 32u) + threadIdx.x / 32u;
    if (row >= out_dim) return;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; ++slot) {
        int32_t si = selected[(uint64_t)tok * n_expert + slot];
        if (si < 0) si = 0;
        const char *db = down_slots ? down_slots[(uint32_t)si] : down_base + (uint64_t)(uint32_t)si * expert_bytes;
        const uint8_t *wr = (const uint8_t *)db + (uint64_t)row * row_bytes;
        const float *xp = mid + ((uint64_t)tok * n_expert + slot) * mid_dim;
        float acc = 0.0f;
        for (uint32_t base = 0; base < mid_dim / 32u; base += 8u) {
            const uint32_t block = base + lane / 4u;
            if (block >= mid_dim / 32u) continue;
            const uint8_t *p = wr + (uint64_t)block * 14u;
            const uint32_t j0 = (lane & 3u) * 8u;
            const uint32_t byte = (lane & 3u) * 3u;
            const uint32_t packed = (uint32_t)p[byte] | ((uint32_t)p[byte + 1u] << 8u) |
                                    ((uint32_t)p[byte + 2u] << 16u);
            const float scale = dev_rocmfpx_ue4m3(p[12u + j0 / 16u]);
            #pragma unroll
            for (uint32_t j = 0; j < 8u; j++) {
                const uint32_t code = (packed >> (j * 3u)) & 7u;
                const uint32_t mc = code & 3u;
                const float value = (code & 4u ? -1.0f : 1.0f) * (float)(mc == 3u ? 4u : mc);
                acc += value * scale * xp[block * 32u + j0 + j];
            }
        }
        total += acc;
    }
    total = warp_sum_f32(total);
    if (lane == 0u) out[(uint64_t)tok * out_dim + row] = total;
}

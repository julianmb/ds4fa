extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || n_vocab == 0u || token >= n_vocab || n_embd == 0u || n_hc == 0u ||
        (uint64_t)n_embd * n_hc > UINT32_MAX) return 0;
    uint64_t out_bytes = 0;
    if (!cuda_u64_mul3_checked(n_hc, n_embd, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(out_hc, out_bytes)) return 0;

    // Check Q4_K first
    const uint64_t q4_k_row_bytes = ((uint64_t)n_embd + 255u) / 256u * 144u;
    uint64_t q4_k_bytes = 0;
    if (cuda_u64_mul_checked(n_vocab, q4_k_row_bytes, &q4_k_bytes) &&
        cuda_model_range_fits(model_size, weight_offset, q4_k_bytes)) {
        const char *wptr = cuda_model_range_ptr(model_map, weight_offset, q4_k_bytes, "token_embd_q4_k");
        if (!wptr) return 0;
        uint64_t n = (uint64_t)n_embd * n_hc;
        embed_token_hc_q4_k_kernel<<<(n + 255u) / 256u, 256>>>((float *)out_hc->ptr, (const unsigned char *)wptr, token, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed token q4_k launch");
    }

    // Check Q8_0
    const uint64_t q8_0_row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    uint64_t q8_0_bytes = 0;
    if (cuda_u64_mul_checked(n_vocab, q8_0_row_bytes, &q8_0_bytes) &&
        cuda_model_range_fits(model_size, weight_offset, q8_0_bytes)) {
        const char *wptr = cuda_model_range_ptr(model_map, weight_offset, q8_0_bytes, "token_embd_q8_0");
        if (!wptr) return 0;
        uint64_t n = (uint64_t)n_embd * n_hc;
        embed_token_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>((float *)out_hc->ptr, (const unsigned char *)wptr, token, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed token q8_0 launch");
    }

    // Fall back to F16
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul3_checked(n_vocab, n_embd, sizeof(uint16_t), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255u) / 256u, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map || n_vocab == 0u || n_tokens == 0u || n_embd == 0u || n_hc == 0u) {
        return 0;
    }
    uint64_t n = 0;
    if (!cuda_tensor_has_i32(tokens_t, n_tokens) ||
        !cuda_u64_mul_checked(n_tokens, n_hc, &n) ||
        !cuda_u64_mul_checked(n, n_embd, &n) ||
        !cuda_tensor_has_f32(out_hc, n)) {
        return 0;
    }

    // Check Q4_K first
    const uint64_t q4_k_row_bytes = ((uint64_t)n_embd + 255u) / 256u * 144u;
    uint64_t q4_k_bytes = 0;
    if (cuda_u64_mul_checked(n_vocab, q4_k_row_bytes, &q4_k_bytes) &&
        cuda_model_range_fits(model_size, weight_offset, q4_k_bytes)) {
        const char *wptr = cuda_model_range_ptr(model_map, weight_offset, q4_k_bytes, "token_embd_q4_k");
        if (!wptr) return 0;
        embed_tokens_hc_q4_k_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const int32_t *)tokens_t->ptr,
            (const unsigned char *)wptr,
            n_vocab, n_tokens, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed tokens q4_k launch");
    }

    // Check Q8_0
    const uint64_t q8_0_row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    uint64_t q8_0_bytes = 0;
    if (cuda_u64_mul_checked(n_vocab, q8_0_row_bytes, &q8_0_bytes) &&
        cuda_model_range_fits(model_size, weight_offset, q8_0_bytes)) {
        const char *wptr = cuda_model_range_ptr(model_map, weight_offset, q8_0_bytes, "token_embd_q8_0");
        if (!wptr) return 0;
        embed_tokens_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const int32_t *)tokens_t->ptr,
            (const unsigned char *)wptr,
            n_vocab, n_tokens, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed tokens q8_0 launch");
    }

    // Fall back to F16
    uint64_t weight_bytes = 0;
    if (!cuda_u64_mul3_checked(n_vocab, n_embd, sizeof(uint16_t), &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    embed_tokens_hc_kernel<<<(n + 255u) / 256u, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

extern "C" int ds4_gpu_embed_token_hc_q8_0_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc) {
    if (!out_hc || !model_map || n_vocab == 0u || token >= n_vocab || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t blocks = (n_embd + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, out_bytes = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(n_vocab, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_u64_mul3_checked(n_embd, n_hc, sizeof(float), &out_bytes) ||
        !cuda_tensor_has_bytes(out_hc, out_bytes)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd_q8_0");
    if (!wptr) return 0;
    const uint64_t n = (uint64_t)n_embd * n_hc;
    embed_token_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const unsigned char *)wptr,
            token,
            n_embd,
            n_hc);
    return cuda_ok(cudaGetLastError(), "embed token q8_0 launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map || n_vocab == 0u || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t blocks = (n_embd + 31u) / 32u;
    uint64_t row_bytes = 0, weight_bytes = 0, n = 0;
    if (!cuda_u64_mul_checked(blocks, 34u, &row_bytes) ||
        !cuda_u64_mul_checked(n_vocab, row_bytes, &weight_bytes) ||
        !cuda_model_range_fits(model_size, weight_offset, weight_bytes) ||
        !cuda_tensor_has_i32(tokens_t, n_tokens) ||
        !cuda_u64_mul_checked(n_tokens, n_hc, &n) ||
        !cuda_u64_mul_checked(n, n_embd, &n) ||
        !cuda_tensor_has_f32(out_hc, n)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd_q8_0");
    if (!wptr) return 0;
    embed_tokens_hc_q8_0_kernel<<<(n + 255u) / 256u, 256>>>(
            (float *)out_hc->ptr,
            (const int32_t *)tokens_t->ptr,
            (const unsigned char *)wptr,
            n_vocab,
            n_tokens,
            n_embd,
            n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens q8_0 launch");
}

extern "C" int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd) {
    return ds4_gpu_embed_token_hc_q8_0_tensor(out,
                                              model_map,
                                              model_size,
                                              weight_offset,
                                              n_vocab,
                                              token,
                                              n_embd,
                                              1);
}

extern "C" int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd) {
    return ds4_gpu_embed_tokens_hc_q8_0_tensor(out,
                                               tokens_t,
                                               model_map,
                                               model_size,
                                               weight_offset,
                                               n_vocab,
                                               n_tokens,
                                               n_embd,
                                               1);
}

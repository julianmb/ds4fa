#ifndef DS4_HIP_H
#define DS4_HIP_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_hip_tensor ds4_hip_tensor;

int ds4_hip_init(void);
void ds4_hip_cleanup(void);

ds4_hip_tensor *ds4_hip_tensor_alloc(uint64_t bytes);
ds4_hip_tensor *ds4_hip_tensor_view(const ds4_hip_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_hip_tensor_free(ds4_hip_tensor *tensor);
uint64_t ds4_hip_tensor_bytes(const ds4_hip_tensor *tensor);
void *ds4_hip_tensor_contents(ds4_hip_tensor *tensor);
int ds4_hip_tensor_write(ds4_hip_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_hip_tensor_read(const ds4_hip_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_hip_tensor_copy(ds4_hip_tensor *dst, uint64_t dst_offset,
                        const ds4_hip_tensor *src, uint64_t src_offset,
                        uint64_t bytes);

int ds4_hip_begin_commands(void);
int ds4_hip_flush_commands(void);
int ds4_hip_end_commands(void);
int ds4_hip_synchronize(void);

int ds4_hip_set_model_map(const void *model_map, uint64_t model_size);
int ds4_hip_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
void ds4_hip_set_quality(bool quality);
void ds4_hip_print_memory_report(const char *label);

/* Embeddings and Indexer */
int ds4_hip_embed_token_hc_tensor(
        ds4_hip_tensor *out_hc,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        uint32_t        n_vocab,
        uint32_t        token,
        uint32_t        n_embd,
        uint32_t        n_hc);

int ds4_hip_embed_tokens_hc_tensor(
        ds4_hip_tensor       *out_hc,
        const ds4_hip_tensor *tokens,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_vocab,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_indexer_score_one_tensor(
        ds4_hip_tensor       *scores,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *weights,
        const ds4_hip_tensor *index_comp,
        uint32_t              n_comp,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale);

int ds4_hip_indexer_scores_prefill_tensor(
        ds4_hip_tensor       *scores,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *weights,
        const ds4_hip_tensor *index_comp,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              ratio,
        float                 scale);

int ds4_hip_indexer_scores_decode_batch_tensor(
        ds4_hip_tensor       *scores,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *weights,
        const ds4_hip_tensor *index_comp,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              ratio,
        float                 scale);

int ds4_hip_indexer_topk_tensor(
        ds4_hip_tensor       *selected,
        const ds4_hip_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              top_k);

int ds4_hip_dsv4_topk_mask_tensor(
        ds4_hip_tensor       *mask,
        const ds4_hip_tensor *topk,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              top_k);

/* Matmul, Norm, RoPE */
int ds4_hip_matmul_q8_0_tensor(
        ds4_hip_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x,
        uint64_t              n_tok);

int ds4_hip_shared_gate_up_swiglu_q8_0_tensor(
        ds4_hip_tensor       *gate,
        ds4_hip_tensor       *up,
        ds4_hip_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x);

int ds4_hip_matmul_f16_tensor(
        ds4_hip_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x,
        uint64_t              n_tok);

int ds4_hip_matmul_f16_pair_tensor(
        ds4_hip_tensor       *out_a,
        ds4_hip_tensor       *out_b,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_a_offset,
        uint64_t              weight_b_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x,
        uint64_t              n_tok);

int ds4_hip_matmul_f32_tensor(
        ds4_hip_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x,
        uint64_t              n_tok);

int ds4_hip_repeat_hc_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *row,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_rms_norm_plain_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *x,
        uint32_t              n,
        float                 eps);

int ds4_hip_rms_norm_plain_rows_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *x,
        uint32_t              n,
        uint32_t              rows,
        float                 eps);

int ds4_hip_rms_norm_weight_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        float                 eps);

int ds4_hip_rms_norm_weight_rows_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              rows,
        float                 eps);

int ds4_hip_dsv4_qkv_rms_norm_rows_tensor(
        ds4_hip_tensor       *q_out,
        const ds4_hip_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_hip_tensor       *kv_out,
        const ds4_hip_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        uint32_t              rows,
        float                 eps);

int ds4_hip_head_rms_norm_tensor(
        ds4_hip_tensor *x,
        uint32_t        n_tok,
        uint32_t        n_head,
        uint32_t        head_dim,
        float           eps);

int ds4_hip_dsv4_fp8_kv_quantize_tensor(
        ds4_hip_tensor *x,
        uint32_t        n_tok,
        uint32_t        head_dim,
        uint32_t        n_rot);

int ds4_hip_rope_tail_tensor(
        ds4_hip_tensor *x,
        uint32_t        n_tok,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        bool            inverse,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_hip_kv_fp8_store_raw_tensor(
        ds4_hip_tensor *kv,
        ds4_hip_tensor *raw_cache,
        uint32_t        raw_cap,
        uint32_t        row,
        uint32_t        head_dim,
        uint32_t        n_rot);

int ds4_hip_store_raw_kv_tensor(
        ds4_hip_tensor       *raw_cache,
        const ds4_hip_tensor *kv,
        uint32_t              raw_cap,
        uint32_t              row,
        uint32_t              head_dim);

int ds4_hip_store_raw_kv_batch_tensor(
        ds4_hip_tensor       *raw_cache,
        const ds4_hip_tensor *kv,
        uint32_t              raw_cap,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              head_dim);

/* KV Compression and Attention */
int ds4_hip_compressor_update_tensor(
        const ds4_hip_tensor *kv_cur,
        const ds4_hip_tensor *sc_cur,
        ds4_hip_tensor       *state_kv,
        ds4_hip_tensor       *state_score,
        ds4_hip_tensor       *comp_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ape_offset,
        uint32_t              ape_type,
        uint64_t              norm_offset,
        uint32_t              norm_type,
        uint32_t              head_dim,
        uint32_t              ratio,
        uint32_t              pos,
        uint32_t              comp_row,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

int ds4_hip_compressor_store_batch_tensor(
        const ds4_hip_tensor *kv,
        const ds4_hip_tensor *sc,
        ds4_hip_tensor       *state_kv,
        ds4_hip_tensor       *state_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ape_offset,
        uint32_t              ape_type,
        uint32_t              head_dim,
        uint32_t              ratio,
        uint32_t              pos0,
        uint32_t              n_tokens);

int ds4_hip_compressor_prefill_tensor(
        ds4_hip_tensor       *comp_cache,
        ds4_hip_tensor       *state_kv,
        ds4_hip_tensor       *state_score,
        const ds4_hip_tensor *kv,
        const ds4_hip_tensor *sc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ape_offset,
        uint32_t              ape_type,
        uint64_t              norm_offset,
        uint32_t              norm_type,
        uint32_t              head_dim,
        uint32_t              ratio,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        bool                  quantize_fp8,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

int ds4_hip_compressor_prefill_ratio4_replay_tensor(
        ds4_hip_tensor       *comp_cache,
        ds4_hip_tensor       *state_kv,
        ds4_hip_tensor       *state_score,
        const ds4_hip_tensor *kv,
        const ds4_hip_tensor *sc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ape_offset,
        uint32_t              ape_type,
        uint64_t              norm_offset,
        uint32_t              norm_type,
        uint32_t              head_dim,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        bool                  quantize_fp8,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

int ds4_hip_compressor_prefill_state_ratio4_tensor(
        ds4_hip_tensor       *state_kv,
        ds4_hip_tensor       *state_score,
        const ds4_hip_tensor *kv_tail,
        const ds4_hip_tensor *sc_tail,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ape_offset,
        uint32_t              ape_type,
        uint32_t              head_dim,
        uint32_t              pos0);

int ds4_hip_attention_decode_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        const ds4_hip_tensor *comp_kv,
        uint32_t              n_comp,
        const ds4_hip_tensor *comp_mask,
        uint32_t              use_mask,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_prefill_raw_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        uint32_t              n_tokens,
        uint32_t              window,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_decode_raw_batch_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              window,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_decode_mixed_batch_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        const ds4_hip_tensor *comp_kv,
        const ds4_hip_tensor *comp_mask,
        uint32_t              use_comp_mask,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_indexed_mixed_batch_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        const ds4_hip_tensor *comp_kv,
        const ds4_hip_tensor *topk,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_raw,
        uint32_t              raw_cap,
        uint32_t              raw_start,
        uint32_t              n_comp,
        uint32_t              top_k,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_prefill_static_mixed_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        const ds4_hip_tensor *comp_kv,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_prefill_masked_mixed_heads_tensor(
        ds4_hip_tensor       *heads,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              sinks_offset,
        const ds4_hip_tensor *q,
        const ds4_hip_tensor *raw_kv,
        const ds4_hip_tensor *comp_kv,
        const ds4_hip_tensor *comp_mask,
        uint32_t              n_tokens,
        uint32_t              n_comp,
        uint32_t              window,
        uint32_t              ratio,
        uint32_t              n_head,
        uint32_t              head_dim);

int ds4_hip_attention_output_q8_batch_tensor(
        ds4_hip_tensor       *out,
        ds4_hip_tensor       *low,
        ds4_hip_tensor       *group_tmp,
        ds4_hip_tensor       *low_tmp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              out_a_offset,
        uint64_t              out_b_offset,
        uint64_t              group_dim,
        uint64_t              rank,
        uint32_t              n_groups,
        uint64_t              out_dim,
        const ds4_hip_tensor *heads,
        uint32_t              n_tokens);

int ds4_hip_attention_output_low_q8_tensor(
        ds4_hip_tensor       *low,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              out_a_offset,
        uint64_t              group_dim,
        uint64_t              rank,
        uint32_t              n_groups,
        const ds4_hip_tensor *heads);

/* Router, MoE */
int ds4_hip_swiglu_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *gate,
        const ds4_hip_tensor *up,
        uint32_t              n,
        float                 clamp,
        float                 weight);

int ds4_hip_add_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *a,
        const ds4_hip_tensor *b,
        uint32_t              n);

int ds4_hip_router_select_tensor(
        ds4_hip_tensor       *selected,
        ds4_hip_tensor       *weights,
        ds4_hip_tensor       *probs,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        uint64_t              hash_offset,
        uint32_t              hash_rows,
        uint32_t              token,
        uint32_t              n_expert_groups,
        uint32_t              n_group_used,
        bool                  has_bias,
        bool                  hash_mode,
        const ds4_hip_tensor *logits);

int ds4_hip_router_select_batch_tensor(
        ds4_hip_tensor       *selected,
        ds4_hip_tensor       *weights,
        ds4_hip_tensor       *probs,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              bias_offset,
        uint64_t              hash_offset,
        uint32_t              hash_rows,
        uint32_t              n_expert_groups,
        uint32_t              n_group_used,
        bool                  has_bias,
        bool                  hash_mode,
        const ds4_hip_tensor *logits,
        const ds4_hip_tensor *tokens,
        uint32_t              n_tokens);

int ds4_hip_routed_moe_one_tensor(
        ds4_hip_tensor       *out,
        ds4_hip_tensor       *gate,
        ds4_hip_tensor       *up,
        ds4_hip_tensor       *mid,
        ds4_hip_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_hip_tensor *selected,
        const ds4_hip_tensor *weights,
        uint32_t              n_expert,
        float                 clamp,
        const ds4_hip_tensor *x);

int ds4_hip_routed_moe_batch_tensor(
        ds4_hip_tensor       *out,
        ds4_hip_tensor       *gate,
        ds4_hip_tensor       *up,
        ds4_hip_tensor       *mid,
        ds4_hip_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_hip_tensor *selected,
        const ds4_hip_tensor *weights,
        uint32_t              n_expert,
        float                 clamp,
        const ds4_hip_tensor *x,
        uint32_t              n_tokens);

/* Hyper-Connection */
int ds4_hip_hc_split_sinkhorn_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *mix,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps);

int ds4_hip_hc_weighted_sum_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_hc_weighted_sum_split_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_hc_split_weighted_sum_tensor(
        ds4_hip_tensor       *out,
        ds4_hip_tensor       *split,
        const ds4_hip_tensor *mix,
        const ds4_hip_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps);

int ds4_hip_hc_split_weighted_sum_norm_tensor(
        ds4_hip_tensor       *out,
        ds4_hip_tensor       *norm_out,
        ds4_hip_tensor       *split,
        const ds4_hip_tensor *mix,
        const ds4_hip_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 norm_eps);

int ds4_hip_output_hc_weights_tensor(
        ds4_hip_tensor       *out,
        const ds4_hip_tensor *pre,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint32_t              n_hc,
        float                 eps);

int ds4_hip_hc_expand_tensor(
        ds4_hip_tensor       *out_hc,
        const ds4_hip_tensor *block_out,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *post,
        const ds4_hip_tensor *comb,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_hc_expand_split_tensor(
        ds4_hip_tensor       *out_hc,
        const ds4_hip_tensor *block_out,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_hc_expand_add_split_tensor(
        ds4_hip_tensor       *out_hc,
        const ds4_hip_tensor *block_out,
        const ds4_hip_tensor *block_add,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_shared_down_hc_expand_q8_0_tensor(
        ds4_hip_tensor       *out_hc,
        ds4_hip_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *shared_mid,
        const ds4_hip_tensor *routed_out,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_hip_matmul_q8_0_hc_expand_tensor(
        ds4_hip_tensor       *out_hc,
        ds4_hip_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_hip_tensor *x,
        const ds4_hip_tensor *residual_hc,
        const ds4_hip_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

#ifdef __cplusplus
}
#endif

#endif

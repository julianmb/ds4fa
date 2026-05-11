#ifndef DS4_GPU_H
#define DS4_GPU_H

#if defined(DS4_USE_ROCM)

#include "ds4_hip.h"

#define ds4_metal_tensor                     ds4_hip_tensor
#define ds4_metal_init                       ds4_hip_init
#define ds4_metal_cleanup                    ds4_hip_cleanup
#define ds4_metal_tensor_alloc               ds4_hip_tensor_alloc
#define ds4_metal_tensor_view                ds4_hip_tensor_view
#define ds4_metal_tensor_free                ds4_hip_tensor_free
#define ds4_metal_tensor_bytes               ds4_hip_tensor_bytes
#define ds4_metal_tensor_contents            ds4_hip_tensor_contents
#define ds4_metal_tensor_write               ds4_hip_tensor_write
#define ds4_metal_tensor_read                ds4_hip_tensor_read
#define ds4_metal_tensor_copy                ds4_hip_tensor_copy
#define ds4_metal_begin_commands             ds4_hip_begin_commands
#define ds4_metal_flush_commands             ds4_hip_flush_commands
#define ds4_metal_end_commands               ds4_hip_end_commands
#define ds4_metal_synchronize                ds4_hip_synchronize
#define ds4_metal_set_model_map              ds4_hip_set_model_map
#define ds4_metal_set_model_map_range        ds4_hip_set_model_map_range
#define ds4_metal_set_quality                ds4_hip_set_quality
#define ds4_metal_print_memory_report        ds4_hip_print_memory_report

#define ds4_metal_embed_token_hc_tensor      ds4_hip_embed_token_hc_tensor
#define ds4_metal_embed_tokens_hc_tensor     ds4_hip_embed_tokens_hc_tensor
#define ds4_metal_indexer_score_one_tensor   ds4_hip_indexer_score_one_tensor
#define ds4_metal_indexer_scores_prefill_tensor ds4_hip_indexer_scores_prefill_tensor
#define ds4_metal_indexer_scores_decode_batch_tensor ds4_hip_indexer_scores_decode_batch_tensor
#define ds4_metal_indexer_topk_tensor        ds4_hip_indexer_topk_tensor
#define ds4_metal_dsv4_topk_mask_tensor      ds4_hip_dsv4_topk_mask_tensor

#define ds4_metal_matmul_q8_0_tensor         ds4_hip_matmul_q8_0_tensor
#define ds4_metal_shared_gate_up_swiglu_q8_0_tensor ds4_hip_shared_gate_up_swiglu_q8_0_tensor
#define ds4_metal_matmul_f16_tensor          ds4_hip_matmul_f16_tensor
#define ds4_metal_matmul_f16_pair_tensor     ds4_hip_matmul_f16_pair_tensor
#define ds4_metal_matmul_f32_tensor          ds4_hip_matmul_f32_tensor
#define ds4_metal_repeat_hc_tensor           ds4_hip_repeat_hc_tensor
#define ds4_metal_rms_norm_plain_tensor      ds4_hip_rms_norm_plain_tensor
#define ds4_metal_rms_norm_plain_rows_tensor ds4_hip_rms_norm_plain_rows_tensor
#define ds4_metal_rms_norm_weight_tensor     ds4_hip_rms_norm_weight_tensor
#define ds4_metal_rms_norm_weight_rows_tensor ds4_hip_rms_norm_weight_rows_tensor
#define ds4_metal_dsv4_qkv_rms_norm_rows_tensor ds4_hip_dsv4_qkv_rms_norm_rows_tensor
#define ds4_metal_head_rms_norm_tensor       ds4_hip_head_rms_norm_tensor
#define ds4_metal_dsv4_fp8_kv_quantize_tensor ds4_hip_dsv4_fp8_kv_quantize_tensor
#define ds4_metal_rope_tail_tensor           ds4_hip_rope_tail_tensor
#define ds4_metal_kv_fp8_store_raw_tensor    ds4_hip_kv_fp8_store_raw_tensor
#define ds4_metal_store_raw_kv_tensor        ds4_hip_store_raw_kv_tensor
#define ds4_metal_store_raw_kv_batch_tensor  ds4_hip_store_raw_kv_batch_tensor

#define ds4_metal_compressor_update_tensor   ds4_hip_compressor_update_tensor
#define ds4_metal_compressor_store_batch_tensor ds4_hip_compressor_store_batch_tensor
#define ds4_metal_compressor_prefill_tensor  ds4_hip_compressor_prefill_tensor
#define ds4_metal_compressor_prefill_ratio4_replay_tensor ds4_hip_compressor_prefill_ratio4_replay_tensor
#define ds4_metal_compressor_prefill_state_ratio4_tensor ds4_hip_compressor_prefill_state_ratio4_tensor

#define ds4_metal_attention_decode_heads_tensor ds4_hip_attention_decode_heads_tensor
#define ds4_metal_attention_prefill_raw_heads_tensor ds4_hip_attention_prefill_raw_heads_tensor
#define ds4_metal_attention_decode_raw_batch_heads_tensor ds4_hip_attention_decode_raw_batch_heads_tensor
#define ds4_metal_attention_decode_mixed_batch_heads_tensor ds4_hip_attention_decode_mixed_batch_heads_tensor
#define ds4_metal_attention_indexed_mixed_batch_heads_tensor ds4_hip_attention_indexed_mixed_batch_heads_tensor
#define ds4_metal_attention_prefill_static_mixed_heads_tensor ds4_hip_attention_prefill_static_mixed_heads_tensor
#define ds4_metal_attention_prefill_masked_mixed_heads_tensor ds4_hip_attention_prefill_masked_mixed_heads_tensor
#define ds4_metal_attention_output_q8_batch_tensor ds4_hip_attention_output_q8_batch_tensor
#define ds4_metal_attention_output_low_q8_tensor ds4_hip_attention_output_low_q8_tensor

#define ds4_metal_swiglu_tensor              ds4_hip_swiglu_tensor
#define ds4_metal_add_tensor                 ds4_hip_add_tensor
#define ds4_metal_router_select_tensor       ds4_hip_router_select_tensor
#define ds4_metal_router_select_batch_tensor ds4_hip_router_select_batch_tensor
#define ds4_metal_routed_moe_one_tensor      ds4_hip_routed_moe_one_tensor
#define ds4_metal_routed_moe_batch_tensor     ds4_hip_routed_moe_batch_tensor

#define ds4_metal_hc_split_sinkhorn_tensor   ds4_hip_hc_split_sinkhorn_tensor
#define ds4_metal_hc_weighted_sum_tensor     ds4_hip_hc_weighted_sum_tensor
#define ds4_metal_hc_weighted_sum_split_tensor ds4_hip_hc_weighted_sum_split_tensor
#define ds4_metal_hc_split_weighted_sum_tensor ds4_hip_hc_split_weighted_sum_tensor
#define ds4_metal_hc_split_weighted_sum_norm_tensor ds4_hip_hc_split_weighted_sum_norm_tensor
#define ds4_metal_output_hc_weights_tensor   ds4_hip_output_hc_weights_tensor
#define ds4_metal_hc_expand_tensor           ds4_hip_hc_expand_tensor
#define ds4_metal_hc_expand_split_tensor     ds4_hip_hc_expand_split_tensor
#define ds4_metal_hc_expand_add_split_tensor ds4_hip_hc_expand_add_split_tensor
#define ds4_metal_shared_down_hc_expand_q8_0_tensor ds4_hip_shared_down_hc_expand_q8_0_tensor
#define ds4_metal_matmul_q8_0_hc_expand_tensor ds4_hip_matmul_q8_0_hc_expand_tensor

#elif !defined(DS4_NO_METAL)

#include "ds4_metal.h"

#endif

#endif

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2MoE.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct LFM2MoEConfiguration {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_hidden_layers;
    int num_experts;
    int num_experts_per_tok;
    bool norm_topk_prob;
    int num_attention_heads;
    int num_key_value_heads;
    int max_position_embeddings;
    bool use_expert_bias;
    int num_dense_layers;
    float norm_eps;
    bool conv_bias;
    int conv_l_cache;
    float rope_theta;
    std::vector<int> full_attn_idxs;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, LFM2MoEConfiguration& c);

class LFM2MoEAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_, rope_theta_, norm_eps_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;

public:
    explicit LFM2MoEAttention(const LFM2MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoEShortConv {
    int hidden_size_, l_cache_;
    bool bias_;
    mlx::core::array conv_weight_;
    std::optional<mlx::core::array> conv_bias_;
    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

public:
    LFM2MoEShortConv(const LFM2MoEConfiguration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoEMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
public:
    LFM2MoEMLP(int hidden_size, int intermediate_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoESparseMoeBlock {
    int num_experts_, top_k_;
    bool norm_topk_prob_, use_expert_bias_;
    mlx::core::array gate_weight_;
    std::optional<mlx::core::array> expert_bias_;
    SwitchGLU switch_mlp_;

public:
    explicit LFM2MoESparseMoeBlock(const LFM2MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoEDecoderLayer {
    bool is_attention_layer_;
    bool uses_dense_ff_;
    std::optional<LFM2MoEAttention> attention_;
    std::optional<LFM2MoEShortConv> conv_;
    std::optional<LFM2MoEMLP> dense_ff_;
    std::optional<LFM2MoESparseMoeBlock> sparse_ff_;
    mlx::core::array operator_norm_weight_;
    mlx::core::array ffn_norm_weight_;
    float norm_eps_;

public:
    LFM2MoEDecoderLayer(const LFM2MoEConfiguration& config, int layer_idx);
    bool is_attention() const { return is_attention_layer_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoEModelInner {
    mlx::core::array embed_tokens_weight_;
    mlx::core::array embedding_norm_weight_;
    std::vector<LFM2MoEDecoderLayer> layers_;
    float norm_eps_;
    int first_attn_idx_;
    int first_conv_idx_;

public:
    explicit LFM2MoEModelInner(const LFM2MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MoEModel
    : public LanguageModel<LFM2MoEModel>,
      public KVCacheDimensionProvider<LFM2MoEModel> {

    friend class LanguageModel<LFM2MoEModel>;
    friend class KVCacheDimensionProvider<LFM2MoEModel>;

    LFM2MoEConfiguration config_;
    LFM2MoEModelInner model_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit LFM2MoEModel(const LFM2MoEConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

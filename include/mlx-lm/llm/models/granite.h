// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Granite.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct GraniteConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    float logits_scaling;
    float attention_multiplier;
    float embedding_multiplier;
    float residual_multiplier;
    int max_position_embeddings;
    int num_key_value_heads;
    bool attention_bias;
    bool mlp_bias;
    float rope_theta = 10000000.0f;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = true;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, GraniteConfiguration& c);

class GraniteAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_, wo_bias_;
    float rope_theta_;
    float rope_scale_;
public:
    explicit GraniteAttention(const GraniteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GraniteMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    std::optional<mlx::core::array> gate_bias_, down_bias_, up_bias_;
public:
    GraniteMLP(const GraniteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GraniteTransformerBlock {
    GraniteAttention self_attn_;
    GraniteMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
    float residual_multiplier_;
public:
    explicit GraniteTransformerBlock(const GraniteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GraniteModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<GraniteTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    float embedding_multiplier_;
public:
    explicit GraniteModelInner(const GraniteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GraniteModel
    : public LanguageModel<GraniteModel>,
      public KVCacheDimensionProvider<GraniteModel> {

    friend class LanguageModel<GraniteModel>;
    friend class KVCacheDimensionProvider<GraniteModel>;

    GraniteConfiguration config_;
    GraniteModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit GraniteModel(const GraniteConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

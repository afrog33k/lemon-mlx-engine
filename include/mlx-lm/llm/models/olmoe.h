// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of OlmoE.swift
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

struct OlmoEConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;
    int vocab_size;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    bool mlp_bias = false;
    int num_experts;
    int num_experts_per_tok;
    bool norm_topk_prob = false;
    std::optional<int> head_dim;

    int resolved_head_dim() const { return head_dim.value_or(hidden_size / num_attention_heads); }
};

void from_json(const nlohmann::json& j, OlmoEConfiguration& c);

class OlmoEAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    float rope_theta_;
    bool rope_traditional_;
public:
    explicit OlmoEAttention(const OlmoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OlmoESparseMoeBlock {
    int num_experts_, top_k_;
    bool norm_topk_prob_;
    mlx::core::array gate_weight_;
    SwitchGLU switch_mlp_;
public:
    explicit OlmoESparseMoeBlock(const OlmoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OlmoEBlock {
    OlmoEAttention self_attn_;
    OlmoESparseMoeBlock mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float norm_eps_;
public:
    explicit OlmoEBlock(const OlmoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OlmoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<OlmoEBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit OlmoEModelInner(const OlmoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OlmoEModel
    : public LanguageModel<OlmoEModel>,
      public KVCacheDimensionProvider<OlmoEModel> {

    friend class LanguageModel<OlmoEModel>;
    friend class KVCacheDimensionProvider<OlmoEModel>;

    OlmoEConfiguration config_;
    OlmoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit OlmoEModel(const OlmoEConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

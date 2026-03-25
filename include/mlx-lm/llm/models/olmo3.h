// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Olmo3.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx-lm/llm/models/llama.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Olmo3Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    std::optional<int> head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int max_position_embeddings;
    int sliding_window;
    float rope_theta = 10000.0f;
    bool attention_bias = false;
    std::vector<std::string> layer_types; // "full_attention" or "sliding_attention"
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Olmo3Configuration& c);

class Olmo3Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    LlamaDynamicNTKScalingRoPE rope_;
public:
    Olmo3Attention(const Olmo3Configuration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo3MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
public:
    explicit Olmo3MLP(const Olmo3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo3TransformerBlock {
    Olmo3Attention self_attn_;
    Olmo3MLP mlp_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    float rms_norm_eps_;
public:
    Olmo3TransformerBlock(const Olmo3Configuration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo3ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Olmo3TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    std::vector<std::string> layer_types_;
    int sliding_window_;
    int swa_idx_, ga_idx_;
public:
    explicit Olmo3ModelInner(const Olmo3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo3Model
    : public LanguageModel<Olmo3Model>,
      public KVCacheDimensionProvider<Olmo3Model> {

    friend class LanguageModel<Olmo3Model>;
    friend class KVCacheDimensionProvider<Olmo3Model>;

    Olmo3Configuration config_;
    Olmo3ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Olmo3Model(const Olmo3Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

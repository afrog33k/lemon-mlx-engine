// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3Text.swift
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

struct Gemma3TextConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta;
    float rope_local_base_freq;
    bool rope_traditional = false;
    float query_pre_attn_scalar;
    int sliding_window;
    int sliding_window_pattern;
    int max_position_embeddings;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
};

void from_json(const nlohmann::json& j, Gemma3TextConfiguration& c);

// Gemma3 uses: Q/K RMSNorm with +1 offset (Gemma style), GELU activation,
// sliding window pattern, 4 norms per block, query pre-attn scalar for scale,
// embed scaling by sqrt(hidden_size), residual clipping

class Gemma3TextAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    bool is_sliding_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    float rope_theta_; // local or global based on layer
public:
    Gemma3TextAttention(const Gemma3TextConfiguration& config, int layer_idx);
    bool is_sliding() const { return is_sliding_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma3TextMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
public:
    Gemma3TextMLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma3TextTransformerBlock {
    Gemma3TextAttention self_attn_;
    Gemma3TextMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array pre_feedforward_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    float rms_norm_eps_;
public:
    Gemma3TextTransformerBlock(const Gemma3TextConfiguration& config, int layer_idx);
    bool is_sliding() const { return self_attn_.is_sliding(); }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma3TextModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Gemma3TextTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    int hidden_size_;
    int sliding_window_;
    int sliding_window_pattern_;
public:
    explicit Gemma3TextModelInner(const Gemma3TextConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma3TextModel
    : public LanguageModel<Gemma3TextModel> {

    friend class LanguageModel<Gemma3TextModel>;

    Gemma3TextConfiguration config_;
    Gemma3TextModelInner model_;
    mlx::core::array lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Gemma3TextModel(const Gemma3TextConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

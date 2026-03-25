// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Apertus.swift
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

struct ApertusConfiguration {
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    float rms_norm_eps;
    int vocab_size;
    bool tie_word_embeddings = true;
    std::optional<int> max_position_embeddings;
    float rope_theta = 1000000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, ApertusConfiguration& c);

// XIELU activation: learned activation with alpha_p, alpha_n, beta, eps parameters
class XIELU {
    mlx::core::array alpha_p_, alpha_n_, beta_, eps_;
public:
    XIELU();
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class ApertusAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    LlamaDynamicNTKScalingRoPE rope_;
public:
    explicit ApertusAttention(const ApertusConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class ApertusMLP {
    mlx::core::array up_weight_, down_weight_;
    XIELU act_;
public:
    ApertusMLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class ApertusBlock {
    ApertusAttention self_attn_;
    ApertusMLP mlp_;
    mlx::core::array attention_layernorm_weight_;
    mlx::core::array feedforward_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit ApertusBlock(const ApertusConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class ApertusModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<ApertusBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit ApertusModelInner(const ApertusConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class ApertusModel
    : public LanguageModel<ApertusModel>,
      public KVCacheDimensionProvider<ApertusModel> {

    friend class LanguageModel<ApertusModel>;
    friend class KVCacheDimensionProvider<ApertusModel>;

    ApertusConfiguration config_;
    ApertusModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit ApertusModel(const ApertusConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

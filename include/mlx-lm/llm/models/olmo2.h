// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Olmo2.swift
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

struct Olmo2Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    std::optional<int> head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    std::optional<int> max_position_embeddings;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    bool mlp_bias = false;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Olmo2Configuration& c);

class Olmo2Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_, wo_bias_;
    // Q/K RMSNorm weights
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    // RoPE (reuse Llama's implementation)
    LlamaDynamicNTKScalingRoPE rope_;
public:
    explicit Olmo2Attention(const Olmo2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo2MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    std::optional<mlx::core::array> gate_bias_, down_bias_, up_bias_;
public:
    Olmo2MLP(const Olmo2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo2TransformerBlock {
    Olmo2Attention self_attn_;
    Olmo2MLP mlp_;
    // Olmo2 uses post-attention and post-feedforward norms (NOT pre-norms)
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit Olmo2TransformerBlock(const Olmo2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo2ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Olmo2TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit Olmo2ModelInner(const Olmo2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Olmo2Model
    : public LanguageModel<Olmo2Model>,
      public KVCacheDimensionProvider<Olmo2Model> {

    friend class LanguageModel<Olmo2Model>;
    friend class KVCacheDimensionProvider<Olmo2Model>;

    Olmo2Configuration config_;
    Olmo2ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Olmo2Model(const Olmo2Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

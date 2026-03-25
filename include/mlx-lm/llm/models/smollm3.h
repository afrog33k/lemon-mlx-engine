// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SmolLM3.swift
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

struct SmolLM3Configuration {
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
    int no_rope_layer_interval = 4;
    std::vector<int> no_rope_layers; // 1=use rope, 0=no rope

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, SmolLM3Configuration& c);

class SmolLM3Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    bool use_rope_; // per-layer, set by model
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_, wo_bias_;
    float rope_theta_;
    bool rope_traditional_;
public:
    explicit SmolLM3Attention(const SmolLM3Configuration& config);
    void set_use_rope(bool use) { use_rope_ = use; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class SmolLM3MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    std::optional<mlx::core::array> gate_bias_, down_bias_, up_bias_;
public:
    SmolLM3MLP(const SmolLM3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class SmolLM3TransformerBlock {
    SmolLM3Attention self_attn_;
    SmolLM3MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit SmolLM3TransformerBlock(const SmolLM3Configuration& config);
    SmolLM3Attention& attention() { return self_attn_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class SmolLM3ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<SmolLM3TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit SmolLM3ModelInner(const SmolLM3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::vector<SmolLM3TransformerBlock>& layers() { return layers_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class SmolLM3Model
    : public LanguageModel<SmolLM3Model>,
      public KVCacheDimensionProvider<SmolLM3Model> {

    friend class LanguageModel<SmolLM3Model>;
    friend class KVCacheDimensionProvider<SmolLM3Model>;

    SmolLM3Configuration config_;
    SmolLM3ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit SmolLM3Model(const SmolLM3Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

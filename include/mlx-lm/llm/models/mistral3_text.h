// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Mistral3Text.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Mistral3TextConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    std::optional<int> head_dim;
    std::optional<int> max_position_embeddings;
    int num_key_value_heads;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_parameters;
    bool tie_word_embeddings = false;
    std::vector<std::string> layer_types;
    std::optional<int> sliding_window;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Mistral3TextConfiguration& c);

// Llama4-style position-dependent attention scaling
mlx::core::array get_llama4_attention_scale(
    int start, int stop, float beta, int max_position_embeddings, mlx::core::Dtype dtype);

class Mistral3Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    float rope_theta_;

public:
    explicit Mistral3Attention(const Mistral3TextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const mlx::core::array& attn_scale,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Mistral3MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Mistral3MLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Mistral3TextTransformerBlock {
    bool use_sliding_;
    Mistral3Attention attention_;
    Mistral3MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    Mistral3TextTransformerBlock(const Mistral3TextConfiguration& args, bool use_sliding);
    bool use_sliding() const { return use_sliding_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const mlx::core::array& attn_scale,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Mistral3TextModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Mistral3TextTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    std::vector<std::string> layer_types_;
    std::optional<int> sliding_window_;
    int fa_idx_;
    int swa_idx_;

    // Rope parameters for attention scaling
    float llama4_scaling_beta_;
    int original_max_pos_embed_;

public:
    explicit Mistral3TextModelInner(const Mistral3TextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    const std::vector<Mistral3TextTransformerBlock>& get_layers() const { return layers_; }
};

class Mistral3TextModel
    : public LanguageModel<Mistral3TextModel> {

    friend class LanguageModel<Mistral3TextModel>;

    Mistral3TextConfiguration config_;
    Mistral3TextModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Mistral3TextModel(const Mistral3TextConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Ernie4_5.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Ernie45Configuration {
    int hidden_size;
    int intermediate_size;
    int max_position_embeddings;
    int num_attention_heads;
    int num_key_value_heads;
    std::optional<int> head_dim;
    int num_hidden_layers;
    float rms_norm_eps;
    int vocab_size;
    float rope_theta;
    bool use_bias;
    bool tie_word_embeddings;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Ernie45Configuration& c);

class Ernie45Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_, wo_bias_;
    float rope_theta_;
public:
    explicit Ernie45Attention(const Ernie45Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Ernie45MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    std::optional<mlx::core::array> gate_bias_, down_bias_, up_bias_;
public:
    Ernie45MLP(int dim, int hidden_dim, bool use_bias);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Ernie45DecoderLayer {
    Ernie45Attention self_attn_;
    Ernie45MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit Ernie45DecoderLayer(const Ernie45Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Ernie45ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Ernie45DecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit Ernie45ModelInner(const Ernie45Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Ernie45Model
    : public LanguageModel<Ernie45Model>,
      public KVCacheDimensionProvider<Ernie45Model> {

    friend class LanguageModel<Ernie45Model>;
    friend class KVCacheDimensionProvider<Ernie45Model>;

    Ernie45Configuration config_;
    Ernie45ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Ernie45Model(const Ernie45Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4.swift
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

struct GLM4Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    bool attention_bias;
    int head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float partial_rotary_factor;
    float rope_theta = 10000.0f;
    bool rope_traditional = true;
    bool tie_word_embeddings = false;
    int max_position_embeddings = 32768;
};

void from_json(const nlohmann::json& j, GLM4Configuration& c);

class GLM4Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    bool attention_bias_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_;
    float rope_theta_;
    bool rope_traditional_;
    int rope_dims_;
public:
    explicit GLM4Attention(const GLM4Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MLP {
    mlx::core::array gate_up_weight_, down_weight_;
    int intermediate_size_;
public:
    explicit GLM4MLP(const GLM4Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4DecoderLayer {
    GLM4Attention self_attn_;
    GLM4MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array post_self_attn_layernorm_weight_;
    mlx::core::array post_mlp_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit GLM4DecoderLayer(const GLM4Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<GLM4DecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit GLM4ModelInner(const GLM4Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4Model
    : public LanguageModel<GLM4Model>,
      public KVCacheDimensionProvider<GLM4Model> {

    friend class LanguageModel<GLM4Model>;
    friend class KVCacheDimensionProvider<GLM4Model>;

    GLM4Configuration config_;
    GLM4ModelInner model_;
    mlx::core::array lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit GLM4Model(const GLM4Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of MiniCPM.swift
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

struct MiniCPMConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 10000.0f;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    int max_position_embeddings;
    int dim_model_base;
    float scale_depth = 1.0f;
    float scale_emb = 1.0f;
    bool tie_word_embeddings = false;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, MiniCPMConfiguration& c);

class MiniCPMAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    float rope_theta_;
    float rope_scale_;
public:
    explicit MiniCPMAttention(const MiniCPMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiniCPMMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
public:
    MiniCPMMLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiniCPMDecoderLayer {
    MiniCPMAttention self_attn_;
    MiniCPMMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
    float residual_scale_;
public:
    explicit MiniCPMDecoderLayer(const MiniCPMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiniCPMModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<MiniCPMDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    float scale_emb_;
public:
    explicit MiniCPMModelInner(const MiniCPMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiniCPMModel
    : public LanguageModel<MiniCPMModel>,
      public KVCacheDimensionProvider<MiniCPMModel> {

    friend class LanguageModel<MiniCPMModel>;
    friend class KVCacheDimensionProvider<MiniCPMModel>;

    MiniCPMConfiguration config_;
    MiniCPMModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit MiniCPMModel(const MiniCPMConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

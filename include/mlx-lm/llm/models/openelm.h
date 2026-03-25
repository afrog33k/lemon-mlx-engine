// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of OpenELM.swift
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

struct OpenELMConfiguration {
    int head_dim;
    int num_transformer_layers;
    int model_dim;
    int vocab_size;
    int ffn_dim_divisor = 8;
    bool ffn_with_glu = true;
    bool normalize_qk_projections = true;
    bool share_input_output_layers = true;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    int num_gqa_groups = 4;

    // Per-layer computed arrays
    std::vector<int> num_query_heads;
    std::vector<int> kv_heads;
    std::vector<float> ffn_multipliers;
};

void from_json(const nlohmann::json& j, OpenELMConfiguration& c);

class OpenELMAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array qkv_proj_weight_;
    mlx::core::array out_proj_weight_;
    mlx::core::array q_norm_weight_;
    mlx::core::array k_norm_weight_;
    bool has_qk_norm_;
    float norm_eps_;
    float rope_theta_;
    bool rope_traditional_;
public:
    OpenELMAttention(const OpenELMConfiguration& config, int layer_id);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OpenELMFeedForward {
    mlx::core::array proj_1_weight_;
    mlx::core::array proj_2_weight_;
public:
    OpenELMFeedForward(const OpenELMConfiguration& config, int layer_id);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OpenELMBlock {
    OpenELMAttention attn_;
    OpenELMFeedForward ffn_;
    mlx::core::array attn_norm_weight_;
    mlx::core::array ffn_norm_weight_;
    float norm_eps_;
public:
    OpenELMBlock(const OpenELMConfiguration& config, int layer_id);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OpenELMModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<OpenELMBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit OpenELMModelInner(const OpenELMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class OpenELMModel
    : public LanguageModel<OpenELMModel>,
      public KVCacheDimensionProvider<OpenELMModel> {

    friend class LanguageModel<OpenELMModel>;
    friend class KVCacheDimensionProvider<OpenELMModel>;

    OpenELMConfiguration config_;
    OpenELMModelInner transformer_;
    mlx::core::array lm_head_weight_;
    bool has_lm_head_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit OpenELMModel(const OpenELMConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

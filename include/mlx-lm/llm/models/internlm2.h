// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Internlm2.swift
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

struct InternLM2Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;
    bool bias = true;

    int kv_groups() const { return num_attention_heads / num_key_value_heads; }
    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, InternLM2Configuration& c);

// InternLM2 uses: combined QKV (wqkv), dynamic NTK RoPE,
// different weight naming (attention/feed_forward, tok_embeddings, output)

class InternLM2Attention {
    int num_heads_, num_kv_heads_, kv_groups_, head_dim_;
    float scale_;
    mlx::core::array wqkv_weight_;
    std::optional<mlx::core::array> wqkv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;
    // Dynamic NTK scaling RoPE params
    int max_position_embeddings_;
    float rope_theta_;
    bool rope_traditional_;
    float rope_scale_;
public:
    explicit InternLM2Attention(const InternLM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class InternLM2MLP {
    mlx::core::array w1_weight_, w2_weight_, w3_weight_;
public:
    InternLM2MLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class InternLM2TransformerBlock {
    InternLM2Attention attention_;
    InternLM2MLP feed_forward_;
    mlx::core::array attention_norm_weight_;
    mlx::core::array ffn_norm_weight_;
    float rms_norm_eps_;
public:
    explicit InternLM2TransformerBlock(const InternLM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class InternLM2ModelInner {
    mlx::core::array tok_embeddings_weight_;
    std::vector<InternLM2TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit InternLM2ModelInner(const InternLM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class InternLM2Model
    : public LanguageModel<InternLM2Model>,
      public KVCacheDimensionProvider<InternLM2Model> {

    friend class LanguageModel<InternLM2Model>;
    friend class KVCacheDimensionProvider<InternLM2Model>;

    InternLM2Configuration config_;
    InternLM2ModelInner model_;
    std::optional<mlx::core::array> output_weight_; // named "output" not "lm_head"
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit InternLM2Model(const InternLM2Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

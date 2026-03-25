// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of NanoChat.swift
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

struct NanoChatConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    int max_position_embeddings;
    int intermediate_size;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    float logits_softcap = 15.0f;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, NanoChatConfiguration& c);

// NanoChat uses functional RMSNorm (no learned weight), custom RoPE freqs,
// squared relu MLP, logits softcap, and different weight naming (c_q, c_k, c_v, c_proj, c_fc)

class NanoChatAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    float rms_norm_eps_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array rope_freqs_; // precomputed
public:
    explicit NanoChatAttention(const NanoChatConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NanoChatMLP {
    mlx::core::array fc_weight_, proj_weight_;
public:
    explicit NanoChatMLP(const NanoChatConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NanoChatBlock {
    NanoChatAttention attn_;
    NanoChatMLP mlp_;
    float rms_norm_eps_;
public:
    explicit NanoChatBlock(const NanoChatConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NanoChatModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<NanoChatBlock> layers_;
    float rms_norm_eps_;
public:
    explicit NanoChatModelInner(const NanoChatConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NanoChatModel
    : public LanguageModel<NanoChatModel>,
      public KVCacheDimensionProvider<NanoChatModel> {

    friend class LanguageModel<NanoChatModel>;
    friend class KVCacheDimensionProvider<NanoChatModel>;

    NanoChatConfiguration config_;
    NanoChatModelInner transformer_;
    mlx::core::array lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit NanoChatModel(const NanoChatConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

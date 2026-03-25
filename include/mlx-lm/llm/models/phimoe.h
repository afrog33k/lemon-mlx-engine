// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of PhiMoE.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct PhiMoEConfiguration {
    int vocab_size = 32064;
    int hidden_size = 4096;
    int intermediate_size = 6400;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    float rms_norm_eps = 1e-6f;
    int num_local_experts = 16;
    int num_experts_per_tok = 2;
    float rope_theta = 10000.0f;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, PhiMoEConfiguration& c);

class PhiMoEAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_, wo_bias_;
    float rope_theta_;
public:
    explicit PhiMoEAttention(const PhiMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiMoESparseMoeBlock {
    int num_experts_, top_k_;
    mlx::core::array gate_weight_;
    SwitchGLU switch_mlp_;
public:
    explicit PhiMoESparseMoeBlock(const PhiMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiMoEBlock {
    PhiMoEAttention self_attn_;
    PhiMoESparseMoeBlock block_sparse_moe_;
    mlx::core::array input_layernorm_weight_, input_layernorm_bias_;
    mlx::core::array post_attention_layernorm_weight_, post_attention_layernorm_bias_;
    float norm_eps_;
public:
    explicit PhiMoEBlock(const PhiMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiMoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<PhiMoEBlock> layers_;
    mlx::core::array norm_weight_, norm_bias_;
    float norm_eps_;
public:
    explicit PhiMoEModelInner(const PhiMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiMoEModel
    : public LanguageModel<PhiMoEModel>,
      public KVCacheDimensionProvider<PhiMoEModel> {

    friend class LanguageModel<PhiMoEModel>;
    friend class KVCacheDimensionProvider<PhiMoEModel>;

    PhiMoEConfiguration config_;
    PhiMoEModelInner model_;
    mlx::core::array lm_head_weight_, lm_head_bias_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit PhiMoEModel(const PhiMoEConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

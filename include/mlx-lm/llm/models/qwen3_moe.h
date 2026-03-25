// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Qwen3MoE.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/switch_layers.h>
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

struct Qwen3MoEConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_experts;
    int num_experts_per_tok;
    int decoder_sparse_step;
    std::vector<int> mlp_only_layers;
    int moe_intermediate_size;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int head_dim;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = false;
    bool norm_topk_prob = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
};

void from_json(const nlohmann::json& j, Qwen3MoEConfiguration& c);

class Qwen3MoEAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    float rope_theta_;
    float rope_scale_;

public:
    explicit Qwen3MoEAttention(const Qwen3MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3MoEMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Qwen3MoEMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3MoESparseMoeBlock {
    int num_experts_;
    int top_k_;
    bool norm_topk_prob_;

    mlx::core::array gate_weight_;  // [num_experts, hidden_size]
    SwitchGLU switch_mlp_;

public:
    Qwen3MoESparseMoeBlock(const Qwen3MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3MoETransformerBlock {
    Qwen3MoEAttention attention_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    // Either dense MLP or MoE — use a flag to track which
    bool use_moe_;
    std::optional<Qwen3MoEMLP> dense_mlp_;
    std::optional<Qwen3MoESparseMoeBlock> moe_mlp_;

public:
    Qwen3MoETransformerBlock(const Qwen3MoEConfiguration& args, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3MoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen3MoETransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3MoEModelInner(const Qwen3MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3MoEModel
    : public LanguageModel<Qwen3MoEModel>,
      public KVCacheDimensionProvider<Qwen3MoEModel> {

    friend class LanguageModel<Qwen3MoEModel>;
    friend class KVCacheDimensionProvider<Qwen3MoEModel>;

    Qwen3MoEConfiguration config_;
    Qwen3MoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Qwen3MoEModel(const Qwen3MoEConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

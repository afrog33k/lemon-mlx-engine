// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4MOE.swift
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

struct GLM4MoEConfiguration {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;
    float rope_theta;
    float partial_rotary_factor;
    bool use_qk_norm;
    bool tie_word_embeddings;
    bool attention_bias;
    bool norm_topk_prob;
    int n_group;
    int topk_group;
    int num_experts_per_tok;
    float routed_scaling_factor;
    int first_k_dense_replace;
    std::optional<int> n_routed_experts;
    std::optional<int> n_shared_experts;
    std::string scoring_func = "sigmoid";
};

void from_json(const nlohmann::json& j, GLM4MoEConfiguration& c);

class GLM4MoEAttention {
    int num_heads_, num_kv_heads_, head_dim_, rope_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    bool has_qk_norm_;
    float rms_norm_eps_, rope_theta_;
public:
    explicit GLM4MoEAttention(const GLM4MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
public:
    GLM4MoEMLP(int hidden_size, int intermediate_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEGate {
    int top_k_, n_routed_experts_, n_group_, topk_group_;
    bool norm_topk_prob_;
    float routed_scaling_factor_;
    std::string scoring_func_;
    mlx::core::array weight_;
    mlx::core::array e_score_correction_bias_;
public:
    explicit GLM4MoEGate(const GLM4MoEConfiguration& config);
    std::pair<mlx::core::array, mlx::core::array> operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEMoEBlock {
    GLM4MoEGate gate_;
    SwitchGLU switch_mlp_;
    std::optional<GLM4MoEMLP> shared_experts_;
public:
    explicit GLM4MoEMoEBlock(const GLM4MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEBlock {
    GLM4MoEAttention self_attn_;
    bool use_moe_;
    std::optional<GLM4MoEMLP> dense_mlp_;
    std::optional<GLM4MoEMoEBlock> moe_mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float norm_eps_;
public:
    GLM4MoEBlock(const GLM4MoEConfiguration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<GLM4MoEBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit GLM4MoEModelInner(const GLM4MoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoEModel
    : public LanguageModel<GLM4MoEModel>,
      public KVCacheDimensionProvider<GLM4MoEModel> {

    friend class LanguageModel<GLM4MoEModel>;
    friend class KVCacheDimensionProvider<GLM4MoEModel>;

    GLM4MoEConfiguration config_;
    GLM4MoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit GLM4MoEModel(const GLM4MoEConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

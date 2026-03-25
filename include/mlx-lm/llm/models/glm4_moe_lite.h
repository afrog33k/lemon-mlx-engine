// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4MOELite.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
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

struct GLM4MoELiteConfiguration {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    std::optional<int> n_shared_experts;
    std::optional<int> n_routed_experts;
    float routed_scaling_factor;
    int kv_lora_rank;
    std::optional<int> q_lora_rank;
    int qk_rope_head_dim;
    int qk_nope_head_dim;
    int v_head_dim;
    std::string topk_method = "noaux_tc";
    std::string scoring_func = "sigmoid";
    bool norm_topk_prob = true;
    int n_group = 1;
    int topk_group = 1;
    int num_experts_per_tok = 4;
    int moe_layer_freq = 1;
    int first_k_dense_replace = 1;
    int max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool rope_traditional = true;
    bool attention_bias = false;
    int num_nextn_predict_layers = 1;
};

void from_json(const nlohmann::json& j, GLM4MoELiteConfiguration& c);

class GLM4MoELiteAttention {
    int num_heads_, qk_rope_head_dim_, kv_lora_rank_, v_head_dim_;
    int qk_nope_head_dim_, q_head_dim_;
    float scale_;
    bool use_q_lora_;
    float rms_norm_eps_, rope_theta_;

    // Q projections (either direct or LoRA)
    std::optional<mlx::core::array> q_proj_weight_;
    std::optional<mlx::core::array> q_a_proj_weight_, q_a_proj_bias_;
    std::optional<mlx::core::array> q_a_layernorm_weight_;
    std::optional<mlx::core::array> q_b_proj_weight_;

    // KV projections
    mlx::core::array kv_a_proj_weight_;
    std::optional<mlx::core::array> kv_a_proj_bias_;
    mlx::core::array kv_a_layernorm_weight_;
    mlx::core::array kv_b_proj_weight_;

    // Output
    mlx::core::array o_proj_weight_;
    std::optional<mlx::core::array> o_proj_bias_;

public:
    explicit GLM4MoELiteAttention(const GLM4MoELiteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
public:
    GLM4MoELiteMLP(int hidden_size, int intermediate_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteGate {
    int top_k_, n_routed_experts_, n_group_, topk_group_;
    bool norm_topk_prob_;
    float routed_scaling_factor_;
    mlx::core::array weight_;
    mlx::core::array e_score_correction_bias_;
public:
    explicit GLM4MoELiteGate(const GLM4MoELiteConfiguration& config);
    std::pair<mlx::core::array, mlx::core::array> operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteMoE {
    GLM4MoELiteGate gate_;
    SwitchGLU switch_mlp_;
    std::optional<GLM4MoELiteMLP> shared_experts_;
public:
    explicit GLM4MoELiteMoE(const GLM4MoELiteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteBlock {
    GLM4MoELiteAttention self_attn_;
    bool use_moe_;
    std::optional<GLM4MoELiteMLP> dense_mlp_;
    std::optional<GLM4MoELiteMoE> moe_mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float norm_eps_;
public:
    GLM4MoELiteBlock(const GLM4MoELiteConfiguration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<GLM4MoELiteBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit GLM4MoELiteModelInner(const GLM4MoELiteConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GLM4MoELiteModel
    : public LanguageModel<GLM4MoELiteModel>,
      public KVCacheDimensionProvider<GLM4MoELiteModel> {

    friend class LanguageModel<GLM4MoELiteModel>;
    friend class KVCacheDimensionProvider<GLM4MoELiteModel>;

    GLM4MoELiteConfiguration config_;
    GLM4MoELiteModelInner model_;
    mlx::core::array lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit GLM4MoELiteModel(const GLM4MoELiteConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

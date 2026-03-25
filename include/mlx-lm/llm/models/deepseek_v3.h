// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of DeepseekV3.swift
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

struct DeepseekV3Configuration {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    std::optional<int> n_shared_experts;
    std::optional<int> n_routed_experts;
    float routed_scaling_factor = 1.0f;
    int kv_lora_rank;
    int q_lora_rank;
    int qk_rope_head_dim;
    int v_head_dim;
    int qk_nope_head_dim;
    bool norm_topk_prob = false;
    int n_group = 1;
    std::optional<int> topk_group;
    std::optional<int> num_experts_per_tok;
    int moe_layer_freq = 1;
    int first_k_dense_replace = 0;
    int max_position_embeddings = 4096;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool attention_bias = false;
};

void from_json(const nlohmann::json& j, DeepseekV3Configuration& c);

// Yarn RoPE helpers
float yarn_find_correction_dim(float num_rotations, float dim, float base, float max_pos_embed);
std::pair<float, float> yarn_find_correction_range(float low_rot, float high_rot, float dim, float base, float max_pos_embed);
float yarn_get_mscale(float scale, float mscale);
mlx::core::array yarn_linear_ramp_mask(float min_val, float max_val, int dim);

class DeepseekV3YarnRoPE {
    float mscale_;
    int dim_;
    mlx::core::array freqs_;

public:
    DeepseekV3YarnRoPE(int dim, int max_pos_embed = 2048, float base = 10000.0f,
                         float scaling_factor = 1.0f, int original_max_pos = 4096,
                         float beta_fast = 32.0f, float beta_slow = 1.0f,
                         float mscale = 1.0f, float mscale_all_dim = 0.0f);
    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

class DeepseekV3Attention {
    int hidden_size_;
    int num_heads_;
    int qk_rope_head_dim_;
    int kv_lora_rank_;
    int v_head_dim_;
    int qk_nope_head_dim_;
    int q_head_dim_;
    float scale_;
    bool use_q_lora_;

    DeepseekV3YarnRoPE rope_;

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
    explicit DeepseekV3Attention(const DeepseekV3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3MLP {
    mlx::core::array gate_proj_weight_, up_proj_weight_, down_proj_weight_;

public:
    DeepseekV3MLP(int hidden_size, int intermediate_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3MoEGate {
    int n_routed_experts_;
    int n_group_;
    std::optional<int> topk_group_;
    std::optional<int> top_k_;
    bool norm_topk_prob_;
    float routed_scaling_factor_;

    mlx::core::array weight_;
    mlx::core::array e_score_correction_bias_;

public:
    explicit DeepseekV3MoEGate(const DeepseekV3Configuration& config);
    std::pair<mlx::core::array, mlx::core::array> operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3MoE {
    int num_experts_per_tok_;
    SwitchGLU switch_mlp_;
    DeepseekV3MoEGate gate_;
    std::optional<DeepseekV3MLP> shared_experts_;

public:
    explicit DeepseekV3MoE(const DeepseekV3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3DecoderLayer {
    DeepseekV3Attention self_attn_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    bool use_moe_;
    std::optional<DeepseekV3MLP> dense_mlp_;
    std::optional<DeepseekV3MoE> moe_mlp_;

public:
    DeepseekV3DecoderLayer(const DeepseekV3Configuration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<DeepseekV3DecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit DeepseekV3ModelInner(const DeepseekV3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class DeepseekV3Model
    : public LanguageModel<DeepseekV3Model>,
      public KVCacheDimensionProvider<DeepseekV3Model> {

    friend class LanguageModel<DeepseekV3Model>;
    friend class KVCacheDimensionProvider<DeepseekV3Model>;

    DeepseekV3Configuration config_;
    DeepseekV3ModelInner model_;
    mlx::core::array lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit DeepseekV3Model(const DeepseekV3Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

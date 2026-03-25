// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++
// Port of GraniteMoeHybrid.swift
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

struct GraniteMoeHybridConfiguration {
    int vocab_size = 32000;
    int hidden_size = 2048;
    int intermediate_size = 8192;
    int num_hidden_layers = 24;
    int max_position_embeddings = 4096;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    bool attention_bias = false;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 1.0f;
    float logits_scaling = 1.0f;
    float residual_multiplier = 1.0f;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool mlp_bias = false;
    std::string position_embedding_type = "rope";
    bool tie_word_embeddings = true;
    std::vector<std::string> layer_types;

    // MoE params (optional)
    std::optional<int> num_local_experts;
    std::optional<int> num_experts_per_token;
    std::optional<int> shared_intermediate_size;

    // Mamba params (optional)
    std::optional<int> mamba_heads;
    std::optional<int> mamba_head_dim;
    std::optional<bool> mamba_proj_bias;
    std::optional<int> mamba_state_dim;
    std::optional<int> mamba_conv_kernel;
    std::optional<int> mamba_groups;
    std::optional<bool> mamba_conv_bias;

    // Time step limits
    float time_step_min = 0.001f;
    float time_step_max = 100.0f;

    // Derived
    bool use_moe() const { return num_local_experts.value_or(0) > 0; }
};

void from_json(const nlohmann::json& j, GraniteMoeHybridConfiguration& c);

// RMSNorm with optional silu gating (simpler than FalconH1's version)
class GraniteMoeHybridRMSNormGated {
    mlx::core::array weight_;
    float eps_;

public:
    GraniteMoeHybridRMSNormGated(int dims, float eps);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& gate = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Mamba2 SSM mixer
class GraniteMoeHybridMamba2Mixer {
    int num_heads_;
    int hidden_size_;
    int ssm_state_size_;
    int conv_kernel_size_;
    int intermediate_size_;
    int num_groups_;
    int head_dim_;
    int conv_dim_;
    float time_step_min_;
    float time_step_max_;

    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array conv1d_weight_;
    std::optional<mlx::core::array> conv1d_bias_;
    mlx::core::array dt_bias_;
    mlx::core::array A_log_;
    mlx::core::array D_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

    GraniteMoeHybridRMSNormGated norm_;

    mlx::core::array apply_conv(const mlx::core::array& input, MambaCache* mc);

public:
    explicit GraniteMoeHybridMamba2Mixer(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Standard attention with optional RoPE
class GraniteMoeHybridAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;
    bool use_rope_;

    mlx::core::array wq_weight_;
    std::optional<mlx::core::array> wq_bias_;
    mlx::core::array wk_weight_;
    std::optional<mlx::core::array> wk_bias_;
    mlx::core::array wv_weight_;
    std::optional<mlx::core::array> wv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;

    float rope_theta_;

public:
    explicit GraniteMoeHybridAttention(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Top-K gating for MoE routing
class GraniteMoeHybridTopKGating {
    int num_experts_;
    int top_k_;
    mlx::core::array layer_weight_;  // [num_experts, hidden_size]

public:
    GraniteMoeHybridTopKGating(int input_size, int num_experts, int top_k);
    // Returns (indices, gates)
    std::pair<mlx::core::array, mlx::core::array> operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// MoE block with SwitchGLU
class GraniteMoeHybridMoE {
    GraniteMoeHybridTopKGating router_;
    SwitchGLU switch_mlp_;

public:
    explicit GraniteMoeHybridMoE(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Shared expert MLP (used alongside MoE)
class GraniteMoeHybridSharedMLP {
    mlx::core::array input_linear_weight_;
    mlx::core::array output_linear_weight_;
    int shared_intermediate_size_;

public:
    explicit GraniteMoeHybridSharedMLP(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Standard GLU MLP (when not using MoE)
class GraniteMoeHybridMLP {
    mlx::core::array gate_weight_;
    std::optional<mlx::core::array> gate_bias_;
    mlx::core::array up_weight_;
    std::optional<mlx::core::array> up_bias_;
    mlx::core::array down_weight_;
    std::optional<mlx::core::array> down_bias_;

public:
    explicit GraniteMoeHybridMLP(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Per-layer block: either mamba or attention, plus MoE or MLP
class GraniteMoeHybridLayer {
    std::string layer_type_;
    float residual_multiplier_;
    bool use_moe_;

    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    // One of these is active based on layer_type
    std::optional<GraniteMoeHybridAttention> self_attn_;
    std::optional<GraniteMoeHybridMamba2Mixer> mamba_;

    // One of these is active based on use_moe
    std::optional<GraniteMoeHybridMoE> block_sparse_moe_;
    std::optional<GraniteMoeHybridSharedMLP> shared_mlp_;
    std::optional<GraniteMoeHybridMLP> mlp_;

public:
    GraniteMoeHybridLayer(const GraniteMoeHybridConfiguration& config, const std::string& layer_type);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Inner model
class GraniteMoeHybridModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<GraniteMoeHybridLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    float embedding_multiplier_;
    std::optional<int> first_attention_index_;
    std::optional<int> first_mamba_index_;

public:
    explicit GraniteMoeHybridModelInner(const GraniteMoeHybridConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Top-level CRTP model
class GraniteMoeHybridModel : public LanguageModel<GraniteMoeHybridModel> {
    friend class LanguageModel<GraniteMoeHybridModel>;

    GraniteMoeHybridConfiguration config_;
    GraniteMoeHybridModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    float logits_scaling_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit GraniteMoeHybridModel(const GraniteMoeHybridConfiguration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

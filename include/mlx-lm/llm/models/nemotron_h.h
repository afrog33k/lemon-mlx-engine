// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of NemotronH.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// Block type indicator parsed from hybrid_override_pattern
enum class NemotronHBlockType {
    Mamba,      // 'M'
    Attention,  // '*'
    MLP,        // '-'
    MoE,        // 'E'
};

NemotronHBlockType parse_block_type(char c);

struct NemotronHConfiguration {
    int vocab_size = 0;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    float layer_norm_epsilon = 1e-5f;
    float rope_theta = 10000.0f;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    bool attention_bias = false;
    std::optional<int> head_dim;
    bool tie_word_embeddings = false;
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int conv_kernel = 0;
    int n_groups = 1;
    bool mamba_proj_bias = false;
    bool use_conv_bias = true;
    int intermediate_size = 0;
    bool mlp_bias = false;
    int n_routed_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int moe_shared_expert_intermediate_size = 0;
    std::optional<int> n_shared_experts;
    int n_group = 1;
    int topk_group = 1;
    bool norm_topk_prob = true;
    float routed_scaling_factor = 1.0f;
    std::string hybrid_override_pattern;
    float time_step_limit_min = 0.0f;
    float time_step_limit_max = std::numeric_limits<float>::infinity();
};

void from_json(const nlohmann::json& j, NemotronHConfiguration& c);

// --- NemotronHRMSNormGated ---
// RMS norm with optional gate (silu) and per-group normalization
class NemotronHRMSNormGated {
    mlx::core::array weight_;
    float eps_;
    int group_size_;

public:
    NemotronHRMSNormGated(int dims, float eps, int group_size);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& gate = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHMamba2Mixer ---
class NemotronHMamba2Mixer {
    int num_heads_;
    int hidden_size_;
    int ssm_state_size_;
    int conv_kernel_size_;
    int intermediate_size_;
    int num_groups_;
    int head_dim_;
    int conv_dim_;
    int heads_per_group_;
    float time_step_limit_min_;
    float time_step_limit_max_;

    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array conv1d_weight_;
    std::optional<mlx::core::array> conv1d_bias_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;
    mlx::core::array dt_bias_;
    mlx::core::array A_log_;
    mlx::core::array D_;

    NemotronHRMSNormGated norm_;

    mlx::core::array apply_conv(const mlx::core::array& conv_input,
                                 const std::optional<mlx::core::array>& mask,
                                 MambaCache* mc);

public:
    explicit NemotronHMamba2Mixer(const NemotronHConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHAttention ---
// Standard attention without RoPE
class NemotronHAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_;
    std::optional<mlx::core::array> wq_bias_;
    mlx::core::array wk_weight_;
    std::optional<mlx::core::array> wk_bias_;
    mlx::core::array wv_weight_;
    std::optional<mlx::core::array> wv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;

public:
    explicit NemotronHAttention(const NemotronHConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHMLP ---
// Simple MLP with squared ReLU activation
class NemotronHMLP {
    mlx::core::array up_proj_weight_;
    std::optional<mlx::core::array> up_proj_bias_;
    mlx::core::array down_proj_weight_;
    std::optional<mlx::core::array> down_proj_bias_;

public:
    NemotronHMLP(const NemotronHConfiguration& config, int intermediate_size_override = 0);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHMoEGate ---
class NemotronHMoEGate {
    int top_k_;
    int n_group_;
    int topk_group_;
    float routed_scaling_factor_;
    bool norm_topk_prob_;
    int n_routed_experts_;

    mlx::core::array weight_;
    mlx::core::array e_score_correction_bias_;

public:
    explicit NemotronHMoEGate(const NemotronHConfiguration& config);
    std::pair<mlx::core::array, mlx::core::array> operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHSwitchMLP ---
// MoE switch MLP using SwitchLinear with squared ReLU (not SiLU/GLU)
class NemotronHSwitchMLP {
    SwitchLinear fc1_;
    SwitchLinear fc2_;

public:
    NemotronHSwitchMLP(int input_dims, int hidden_dims, int num_experts);
    mlx::core::array operator()(const mlx::core::array& x, const mlx::core::array& indices);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHMoE ---
class NemotronHMoE {
    int num_experts_per_tok_;
    NemotronHMoEGate gate_;
    NemotronHSwitchMLP switch_mlp_;
    std::optional<NemotronHMLP> shared_experts_;

public:
    explicit NemotronHMoE(const NemotronHConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHBlock ---
// Generic block wrapping any mixer type, dispatched by block_type
class NemotronHBlock {
    NemotronHBlockType block_type_;
    mlx::core::array norm_weight_;
    float norm_eps_;

    // Exactly one of these is populated based on block_type
    std::optional<NemotronHMamba2Mixer> mamba_mixer_;
    std::optional<NemotronHAttention> attention_;
    std::optional<NemotronHMLP> mlp_;
    std::optional<NemotronHMoE> moe_;

public:
    NemotronHBlock(const NemotronHConfiguration& config, char block_char);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 const std::optional<mlx::core::array>& ssm_mask,
                                 KVCache* cache);
    NemotronHBlockType block_type() const { return block_type_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- NemotronHBackbone ---
class NemotronHBackbone {
    mlx::core::array embeddings_weight_;
    std::vector<NemotronHBlock> layers_;
    mlx::core::array norm_f_weight_;
    float norm_eps_;

    // Index into the cache vector of the first attention layer
    // (counts how many Mamba layers precede the first Attention layer)
    std::optional<int> first_attention_cache_index_;
    // Index into the cache vector of the first mamba layer
    // (counts how many Attention layers precede the first Mamba layer)
    std::optional<int> first_mamba_cache_index_;

    // The parsed pattern for iteration
    std::string pattern_;

public:
    explicit NemotronHBackbone(const NemotronHConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Expose embeddings for tie_word_embeddings
    const mlx::core::array& embeddings_weight() const { return embeddings_weight_; }
};

// --- NemotronHModel ---
class NemotronHModel : public LanguageModel<NemotronHModel> {
    friend class LanguageModel<NemotronHModel>;

    NemotronHConfiguration config_;
    NemotronHBackbone backbone_;
    std::optional<mlx::core::array> lm_head_weight_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit NemotronHModel(const NemotronHConfiguration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

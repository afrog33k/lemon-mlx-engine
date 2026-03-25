// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of FalconH1.swift
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

struct FalconH1Configuration {
    bool attention_bias = false;
    float attention_in_multiplier = 1.0f;
    float attention_out_multiplier = 1.0f;
    float embedding_multiplier = 1.0f;
    int head_dim = 64;
    int hidden_size = 4096;
    std::optional<int> intermediate_size;
    float key_multiplier = 1.0f;
    float lm_head_multiplier = 1.0f;
    bool mamba_conv_bias = true;
    int mamba_d_conv = 4;
    int mamba_d_head = 64;
    int mamba_d_ssm = 1536;
    int mamba_d_state = 256;
    int mamba_expand = 2;
    int mamba_n_groups = 1;
    int mamba_n_heads = 128;
    bool mamba_norm_before_gate = true;
    bool mamba_proj_bias = false;
    bool mamba_rms_norm = false;
    bool mamba_use_mlp = true;
    int max_position_embeddings = 8192;
    bool mlp_bias = false;
    int mlp_expansion_factor = 8;
    std::vector<float> mlp_multipliers = {1.0f, 1.0f};
    int num_attention_heads = 32;
    int num_hidden_layers = 32;
    int num_key_value_heads = 8;
    bool projectors_bias = false;
    float rms_norm_eps = 1e-5f;
    bool rope_traditional = false;
    std::optional<float> rope_scaling;
    float rope_theta = 100000.0f;
    float ssm_in_multiplier = 1.0f;
    std::vector<float> ssm_multipliers = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float ssm_out_multiplier = 1.0f;
    bool tie_word_embeddings = false;
    int vocab_size = 128000;
};

void from_json(const nlohmann::json& j, FalconH1Configuration& c);

// Compute muP vector from config for in_proj scaling
mlx::core::array compute_mup_vector(const FalconH1Configuration& config);

class FalconH1Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;
    float rope_theta_;
    float rope_scale_;
    bool rope_traditional_;
    bool use_bias_;

    mlx::core::array wq_weight_;
    std::optional<mlx::core::array> wq_bias_;
    mlx::core::array wk_weight_;
    std::optional<mlx::core::array> wk_bias_;
    mlx::core::array wv_weight_;
    std::optional<mlx::core::array> wv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;

public:
    explicit FalconH1Attention(const FalconH1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// RMSNorm with optional gating
class RMSNormGated {
    mlx::core::array weight_;
    float eps_;
    bool norm_before_gate_;

public:
    RMSNormGated(int hidden_size, float eps, bool norm_before_gate);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& gate = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Mamba2 mixer for FalconH1
class FalconH1Mixer {
    int num_heads_;
    int hidden_size_;
    int ssm_state_size_;
    int conv_kernel_size_;
    int intermediate_size_;
    bool use_conv_bias_;
    bool use_bias_;
    int groups_time_state_size_;
    int n_groups_;
    int head_dim_;
    int conv_dim_;
    bool mamba_rms_norm_;

    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array conv1d_weight_;
    std::optional<mlx::core::array> conv1d_bias_;
    mlx::core::array dt_bias_;
    mlx::core::array A_log_;
    mlx::core::array D_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

    std::optional<RMSNormGated> norm_;

    float ssm_in_multiplier_;
    float rms_norm_eps_;

    // Apply conv1d with state caching
    mlx::core::array apply_conv(const mlx::core::array& conv_input, MambaCache* mc);

    // SSM step
    std::pair<mlx::core::array, mlx::core::array>
    ssm(const mlx::core::array& hidden_states,
        const mlx::core::array& B,
        const mlx::core::array& C,
        const mlx::core::array& dt,
        const std::optional<mlx::core::array>& state);

public:
    explicit FalconH1Mixer(const FalconH1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x, KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class FalconH1MLP {
    mlx::core::array gate_weight_;
    std::optional<mlx::core::array> gate_bias_;
    mlx::core::array up_weight_;
    std::optional<mlx::core::array> up_bias_;
    mlx::core::array down_weight_;
    std::optional<mlx::core::array> down_bias_;

public:
    explicit FalconH1MLP(const FalconH1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class FalconH1DecoderLayer {
    FalconH1MLP feed_forward_;
    FalconH1Mixer mamba_;
    FalconH1Attention attention_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array pre_ff_layernorm_weight_;
    float norm_eps_;

public:
    explicit FalconH1DecoderLayer(const FalconH1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class FalconH1ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<FalconH1DecoderLayer> layers_;
    mlx::core::array final_layernorm_weight_;
    float norm_eps_;

public:
    explicit FalconH1ModelInner(const FalconH1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class FalconH1Model : public LanguageModel<FalconH1Model> {
    friend class LanguageModel<FalconH1Model>;

    FalconH1Configuration config_;
    FalconH1ModelInner model_;
    mlx::core::array lm_head_weight_;
    mlx::core::array mup_vector_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit FalconH1Model(const FalconH1Configuration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

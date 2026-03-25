// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Jamba.swift
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

struct JambaConfiguration {
    std::string model_type;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int attn_layer_offset;
    int attn_layer_period;
    int expert_layer_offset;
    int expert_layer_period;
    int mamba_d_conv;
    int mamba_d_state;
    int mamba_expand;
    int num_experts;
    int num_experts_per_tok;
    float rms_norm_eps;
    int max_position_embeddings;
    int vocab_size;
    int mamba_dt_rank;
    bool mamba_proj_bias = false;
    bool mamba_conv_bias = true;
    std::vector<std::string> layers_block_type;
    bool tie_word_embeddings = true;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, JambaConfiguration& c);

// Standard SiLU-gated MLP
class JambaMLP {
    mlx::core::array gate_weight_;
    mlx::core::array up_weight_;
    mlx::core::array down_weight_;

public:
    explicit JambaMLP(const JambaConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Standard multi-head attention
class JambaAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_;
    mlx::core::array wk_weight_;
    mlx::core::array wv_weight_;
    mlx::core::array wo_weight_;

public:
    explicit JambaAttention(const JambaConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Mamba SSM mixer with conv1d + SSM step
class JambaMambaMixer {
    int hidden_size_;
    int ssm_state_size_;
    int conv_kernel_size_;
    int intermediate_size_;
    int time_step_rank_;
    bool use_conv_bias_;
    bool use_bias_;

    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;

    // Conv1d as depthwise: weight [intermediate_size, kernel, 1], optional bias
    mlx::core::array conv1d_weight_;
    std::optional<mlx::core::array> conv1d_bias_;

    mlx::core::array x_proj_weight_;

    mlx::core::array dt_proj_weight_;
    mlx::core::array dt_proj_bias_;

    mlx::core::array A_log_;
    mlx::core::array D_;

    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

    // Layer norms for dt, B, C
    mlx::core::array dt_layernorm_weight_;
    mlx::core::array b_layernorm_weight_;
    mlx::core::array c_layernorm_weight_;

    float norm_eps_;

    // SSM step: returns (y, new_ssm_state)
    std::pair<mlx::core::array, mlx::core::array>
    ssm_step(const mlx::core::array& x, const mlx::core::array& A,
             const std::optional<mlx::core::array>& state);

public:
    explicit JambaMambaMixer(const JambaConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x, KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Sparse MoE block with router + SwitchGLU
class JambaSparseMoeBlock {
    int num_experts_per_tok_;
    mlx::core::array router_weight_;
    SwitchGLU switch_mlp_;

public:
    explicit JambaSparseMoeBlock(const JambaConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Decoder layer: attention OR mamba + MLP or MoE
class JambaDecoderLayer {
    bool is_attn_;
    bool is_sparse_moe_;

    std::optional<JambaAttention> self_attn_;
    std::optional<JambaMambaMixer> mamba_;

    std::optional<JambaMLP> mlp_;
    std::optional<JambaSparseMoeBlock> moe_;

    mlx::core::array input_layernorm_weight_;
    mlx::core::array pre_ff_layernorm_weight_;
    float norm_eps_;

public:
    JambaDecoderLayer(const JambaConfiguration& config, const std::string& layer_type);
    bool is_attn() const { return is_attn_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Inner model: embeddings + layers + final norm
class JambaModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<JambaDecoderLayer> layers_;
    mlx::core::array final_layernorm_weight_;
    float norm_eps_;
    int attn_idx_;  // first attention layer index

public:
    explicit JambaModelInner(const JambaConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    const std::vector<JambaDecoderLayer>& get_layers() const { return layers_; }
};

// Top-level Jamba model
class JambaModel : public LanguageModel<JambaModel> {
    friend class LanguageModel<JambaModel>;

    JambaConfiguration config_;
    JambaModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit JambaModel(const JambaConfiguration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

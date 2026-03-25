// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2.swift (non-MoE variant)
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

struct LFM2Configuration {
    int vocab_size;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    float norm_eps;
    bool conv_bias;
    int conv_l_cache;
    int block_dim;         // defaults to hidden_size
    int block_ff_dim;      // defaults to hidden_size
    int block_multiple_of;
    float block_ffn_dim_multiplier;
    bool block_auto_adjust_ff_dim;
    float rope_theta;
    std::vector<int> full_attn_idxs;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, LFM2Configuration& c);

class LFM2Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_, rope_theta_, norm_eps_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;

public:
    explicit LFM2Attention(const LFM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2ShortConv {
    int hidden_size_, l_cache_;
    bool bias_;
    mlx::core::array conv_weight_;
    std::optional<mlx::core::array> conv_bias_;
    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

public:
    LFM2ShortConv(const LFM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2MLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
public:
    LFM2MLP(int hidden_size, int ff_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2DecoderLayer {
    bool is_attention_layer_;
    std::optional<LFM2Attention> attention_;
    std::optional<LFM2ShortConv> conv_;
    LFM2MLP mlp_;
    mlx::core::array operator_norm_weight_;
    mlx::core::array ffn_norm_weight_;
    float norm_eps_;

public:
    LFM2DecoderLayer(const LFM2Configuration& config, int layer_idx);
    bool is_attention() const { return is_attention_layer_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& attn_mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2ModelInner {
    mlx::core::array embed_tokens_weight_;
    mlx::core::array embedding_norm_weight_;
    std::vector<LFM2DecoderLayer> layers_;
    float norm_eps_;
    int first_attn_idx_;

public:
    explicit LFM2ModelInner(const LFM2Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class LFM2Model : public LanguageModel<LFM2Model> {
    friend class LanguageModel<LFM2Model>;

    LFM2Configuration config_;
    LFM2ModelInner model_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit LFM2Model(const LFM2Configuration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

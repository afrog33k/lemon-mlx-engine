// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of BaichuanM1.swift
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

struct BaichuanM1Configuration {
    int vocab_size;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    float rope_theta;
    int sliding_window;
    std::vector<int> sliding_window_layers;
    int conv_window;
    float rms_norm_eps;
    std::optional<int> num_swa_attention_heads;
    std::optional<int> num_swa_key_value_heads;
    bool tie_word_embeddings = false;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, BaichuanM1Configuration& c);

class BaichuanM1Attention {
    int layer_idx_;
    bool is_swa_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;
    float rope_theta_;
    int conv_window_;

    // Combined QKV projection (W_pack)
    mlx::core::array w_pack_weight_;
    mlx::core::array o_proj_weight_;

    // Conv weights for K/V [1, 1, num_kv_heads, 1, conv_window]
    mlx::core::array conv_k_;
    mlx::core::array conv_v_;

    // Custom 2-tap FIR convolution on K or V
    mlx::core::array custom_convolution(const mlx::core::array& u,
                                         const mlx::core::array& weights,
                                         const std::optional<mlx::core::array>& state);

public:
    BaichuanM1Attention(const BaichuanM1Configuration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BaichuanM1MLP {
    mlx::core::array gate_weight_;
    mlx::core::array up_weight_;
    mlx::core::array down_weight_;

public:
    explicit BaichuanM1MLP(const BaichuanM1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BaichuanM1DecoderLayer {
    BaichuanM1Attention attention_;
    BaichuanM1MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float norm_eps_;

public:
    BaichuanM1DecoderLayer(const BaichuanM1Configuration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BaichuanM1ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<BaichuanM1DecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;

public:
    explicit BaichuanM1ModelInner(const BaichuanM1Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BaichuanM1Model : public LanguageModel<BaichuanM1Model> {
    friend class LanguageModel<BaichuanM1Model>;

    BaichuanM1Configuration config_;
    BaichuanM1ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit BaichuanM1Model(const BaichuanM1Configuration& config);
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

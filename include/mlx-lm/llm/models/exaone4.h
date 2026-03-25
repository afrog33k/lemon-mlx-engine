// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Exaone4.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Exaone4Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int max_position_embeddings;
    float rope_theta;
    int head_dim;
    bool tie_word_embeddings;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    std::optional<int> sliding_window;
    std::optional<std::string> sliding_window_pattern;
};

void from_json(const nlohmann::json& j, Exaone4Configuration& c);

class Exaone4Attention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    bool is_local_;
    bool use_rope_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    float rope_theta_;
    float rope_scale_;
public:
    Exaone4Attention(const Exaone4Configuration& config, bool is_local, bool use_rope);
    bool is_local() const { return is_local_; }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Exaone4MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
public:
    Exaone4MLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Exaone4TransformerBlock {
    Exaone4Attention self_attn_;
    Exaone4MLP mlp_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    float rms_norm_eps_;
public:
    Exaone4TransformerBlock(const Exaone4Configuration& config, bool is_local, bool use_rope);
    bool is_local() const { return self_attn_.is_local(); }
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Exaone4ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Exaone4TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit Exaone4ModelInner(const Exaone4Configuration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    const std::vector<Exaone4TransformerBlock>& get_layers() const { return layers_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Exaone4Model
    : public LanguageModel<Exaone4Model>,
      public KVCacheDimensionProvider<Exaone4Model> {

    friend class LanguageModel<Exaone4Model>;
    friend class KVCacheDimensionProvider<Exaone4Model>;

    Exaone4Configuration config_;
    Exaone4ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Exaone4Model(const Exaone4Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

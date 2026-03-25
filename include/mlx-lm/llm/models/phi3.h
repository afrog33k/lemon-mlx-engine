// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
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

struct Phi3RopeScaling {
    std::optional<std::vector<float>> long_factor;
    std::optional<std::vector<float>> short_factor;
    std::optional<float> factor;
    std::optional<std::string> type;
};

void from_json(const nlohmann::json& j, Phi3RopeScaling& c);

struct Phi3Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::optional<Phi3RopeScaling> rope_scaling;
    float partial_rotary_factor = 1.0f;
    int max_position_embeddings;
    int original_max_position_embeddings;
    bool tie_word_embeddings = false;
};

void from_json(const nlohmann::json& j, Phi3Configuration& c);

class Phi3Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int rope_dim_;
    float scale_;

    // Combined QKV projection
    mlx::core::array wqkv_weight_;
    mlx::core::array wo_weight_;

    // RoPE params
    float rope_theta_;
    bool rope_traditional_;
    float rope_scale_;

public:
    explicit Phi3Attention(const Phi3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Phi3MLP {
    mlx::core::array gate_up_weight_;
    mlx::core::array down_weight_;

public:
    Phi3MLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Phi3TransformerBlock {
    Phi3Attention attention_;
    Phi3MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Phi3TransformerBlock(const Phi3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Phi3ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Phi3TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Phi3ModelInner(const Phi3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Phi3Model
    : public LanguageModel<Phi3Model>,
      public KVCacheDimensionProvider<Phi3Model> {

    friend class LanguageModel<Phi3Model>;
    friend class KVCacheDimensionProvider<Phi3Model>;

    Phi3Configuration config_;
    Phi3ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Phi3Model(const Phi3Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

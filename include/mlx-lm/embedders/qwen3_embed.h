// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/embedders/embedding_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Qwen3EmbedConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 1000000.0f;
    int head_dim;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;
};

void from_json(const nlohmann::json& j, Qwen3EmbedConfiguration& c);

class Qwen3EmbedAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;

    // RoPE params
    bool rope_traditional_;
    float rope_theta_;
    float rope_scale_;

public:
    explicit Qwen3EmbedAttention(const Qwen3EmbedConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3EmbedMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Qwen3EmbedMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3EmbedTransformerBlock {
    Qwen3EmbedAttention attention_;
    Qwen3EmbedMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3EmbedTransformerBlock(const Qwen3EmbedConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3EmbedModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen3EmbedTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3EmbedModelInner(const Qwen3EmbedConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3EmbedModel : public EmbeddingModel<Qwen3EmbedModel> {
    friend class EmbeddingModel<Qwen3EmbedModel>;

    Qwen3EmbedConfiguration config_;
    Qwen3EmbedModelInner model_;
    int vocabulary_size_;
    std::vector<int> kv_heads_;

    EmbeddingModelOutput call_impl(const mlx::core::array& inputs,
                                    const std::optional<mlx::core::array>& position_ids,
                                    const std::optional<mlx::core::array>& token_type_ids,
                                    const std::optional<mlx::core::array>& attention_mask);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);
    int vocab_size_impl() const { return vocabulary_size_; }

public:
    explicit Qwen3EmbedModel(const Qwen3EmbedConfiguration& config);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

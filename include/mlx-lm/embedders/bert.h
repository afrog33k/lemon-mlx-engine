// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/embedders/embedding_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct BertConfiguration {
    float layer_norm_eps = 1e-12f;
    int embed_dim = 768;
    int num_heads = 12;
    int inter_dim = 3072;
    int num_layers = 12;
    int type_vocabulary_size = 2;
    int vocabulary_size = 30528;
    int max_position_embeddings = 0;
    std::string model_type;
};

void from_json(const nlohmann::json& j, BertConfiguration& c);

class BertEmbedding {
    mlx::core::array word_embeddings_weight_;
    mlx::core::array norm_weight_, norm_bias_;
    float norm_eps_;
    std::optional<mlx::core::array> token_type_embeddings_weight_;
    mlx::core::array position_embeddings_weight_;
    int type_vocabulary_size_;

public:
    explicit BertEmbedding(const BertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& input_ids,
                                 const std::optional<mlx::core::array>& position_ids = std::nullopt,
                                 const std::optional<mlx::core::array>& token_type_ids = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    const mlx::core::array& word_embeddings_weight() const { return word_embeddings_weight_; }
};

class BertTransformerBlock {
    // Multi-head attention weights
    mlx::core::array query_weight_, query_bias_;
    mlx::core::array key_weight_, key_bias_;
    mlx::core::array value_weight_, value_bias_;
    mlx::core::array out_proj_weight_, out_proj_bias_;
    int num_heads_;
    int head_dim_;

    // Layer norms
    mlx::core::array ln1_weight_, ln1_bias_;
    mlx::core::array ln2_weight_, ln2_bias_;
    float norm_eps_;

    // MLP
    mlx::core::array linear1_weight_, linear1_bias_;
    mlx::core::array linear2_weight_, linear2_bias_;

public:
    explicit BertTransformerBlock(const BertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BertEncoder {
    std::vector<BertTransformerBlock> layers_;

public:
    explicit BertEncoder(const BertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& attention_mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BertLMHead {
    mlx::core::array dense_weight_, dense_bias_;
    mlx::core::array ln_weight_, ln_bias_;
    float norm_eps_;
    mlx::core::array decoder_weight_, decoder_bias_;

public:
    explicit BertLMHead(const BertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BertModel : public EmbeddingModel<BertModel> {
    friend class EmbeddingModel<BertModel>;

    BertConfiguration config_;
    BertEmbedding embedder_;
    BertEncoder encoder_;
    std::optional<BertLMHead> lm_head_;
    std::optional<mlx::core::array> pooler_weight_;
    std::optional<mlx::core::array> pooler_bias_;
    int vocabulary_size_;

    EmbeddingModelOutput call_impl(const mlx::core::array& inputs,
                                    const std::optional<mlx::core::array>& position_ids,
                                    const std::optional<mlx::core::array>& token_type_ids,
                                    const std::optional<mlx::core::array>& attention_mask);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);
    int vocab_size_impl() const { return vocabulary_size_; }

public:
    explicit BertModel(const BertConfiguration& config, bool lm_head = false);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

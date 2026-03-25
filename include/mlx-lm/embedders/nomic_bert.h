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

struct NomicBertConfiguration {
    float layer_norm_eps = 1e-12f;
    bool mlp_fc1_bias = false;
    bool mlp_fc2_bias = false;
    int embed_dim = 768;
    int num_heads = 12;
    int mlp_dim = 3072;
    int num_layers = 12;
    bool qkv_proj_bias = false;
    float rotary_emb_base = 1000.0f;
    float rotary_emb_fraction = 1.0f;
    bool rotary_emb_interleaved = false;
    std::optional<float> rotary_scaling_factor;
    int type_vocabulary_size = 2;
    int vocabulary_size = 30528;
    int max_position_embeddings = 0;
};

void from_json(const nlohmann::json& j, NomicBertConfiguration& c);

class NomicEmbedding {
    mlx::core::array word_embeddings_weight_;
    mlx::core::array norm_weight_, norm_bias_;
    float norm_eps_;
    std::optional<mlx::core::array> token_type_embeddings_weight_;
    std::optional<mlx::core::array> position_embeddings_weight_;
    int type_vocabulary_size_;

public:
    explicit NomicEmbedding(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& input_ids,
                                 const std::optional<mlx::core::array>& position_ids = std::nullopt,
                                 const std::optional<mlx::core::array>& token_type_ids = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    const mlx::core::array& word_embeddings_weight() const { return word_embeddings_weight_; }
};

class NomicMLP {
    mlx::core::array up_weight_, gate_weight_, down_weight_;
    std::optional<mlx::core::array> up_bias_, gate_bias_, down_bias_;

public:
    explicit NomicMLP(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NomicAttention {
    int num_heads_;
    int head_dim_;
    int rotary_emb_dim_;
    float norm_factor_;

    // Combined QKV projection
    mlx::core::array wqkv_weight_;
    std::optional<mlx::core::array> wqkv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;

    // RoPE parameters
    bool use_dynamic_ntk_;
    int dims_;
    std::optional<int> max_position_embeddings_;
    bool traditional_;
    float base_;
    float scale_;

public:
    explicit NomicAttention(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NomicTransformerBlock {
    NomicAttention attention_;
    NomicMLP mlp_;
    mlx::core::array norm1_weight_, norm1_bias_;
    mlx::core::array norm2_weight_, norm2_bias_;
    float norm_eps_;

public:
    explicit NomicTransformerBlock(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NomicEncoder {
    std::vector<NomicTransformerBlock> layers_;

public:
    explicit NomicEncoder(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& attention_mask = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NomicLMHead {
    mlx::core::array dense_weight_, ln_weight_, ln_bias_, decoder_weight_;
    std::optional<mlx::core::array> dense_bias_, decoder_bias_;
    float norm_eps_;

public:
    explicit NomicLMHead(const NomicBertConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class NomicBertModel : public EmbeddingModel<NomicBertModel> {
    friend class EmbeddingModel<NomicBertModel>;

    NomicBertConfiguration config_;
    NomicEmbedding embedder_;
    NomicEncoder encoder_;
    std::optional<NomicLMHead> lm_head_;
    std::optional<mlx::core::array> pooler_weight_;
    int vocabulary_size_;

    EmbeddingModelOutput call_impl(const mlx::core::array& inputs,
                                    const std::optional<mlx::core::array>& position_ids,
                                    const std::optional<mlx::core::array>& token_type_ids,
                                    const std::optional<mlx::core::array>& attention_mask);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);
    int vocab_size_impl() const { return vocabulary_size_; }

public:
    explicit NomicBertModel(const NomicBertConfiguration& config,
                            bool pooler = true, bool lm_head = false);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

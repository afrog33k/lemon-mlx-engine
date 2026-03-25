// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
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

struct CohereConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float layer_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 8000000.0f;
    bool rope_traditional = true;
    float logit_scale;
};

void from_json(const nlohmann::json& j, CohereConfiguration& c);

class CohereAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    float rope_theta_;
    bool rope_traditional_;

public:
    explicit CohereAttention(const CohereConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class CohereMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    CohereMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class CohereTransformerBlock {
    CohereAttention attention_;
    CohereMLP mlp_;
    mlx::core::array input_layernorm_weight_, input_layernorm_bias_;
    float norm_eps_;

public:
    explicit CohereTransformerBlock(const CohereConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class CohereModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<CohereTransformerBlock> layers_;
    mlx::core::array norm_weight_, norm_bias_;
    float norm_eps_;

public:
    explicit CohereModelInner(const CohereConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class CohereModel
    : public LanguageModel<CohereModel>,
      public KVCacheDimensionProvider<CohereModel> {

    friend class LanguageModel<CohereModel>;
    friend class KVCacheDimensionProvider<CohereModel>;

    CohereConfiguration config_;
    CohereModelInner model_;
    float logit_scale_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit CohereModel(const CohereConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

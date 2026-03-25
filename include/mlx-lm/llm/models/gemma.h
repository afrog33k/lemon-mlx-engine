// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/gemma.py
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

struct GemmaConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
};

void from_json(const nlohmann::json& j, GemmaConfiguration& c);

// Gemma uses a custom RMSNorm: rms_norm(x, 1.0 + weight)
class GemmaRMSNorm {
    mlx::core::array weight_;
    float eps_;

public:
    GemmaRMSNorm(int dimensions, float eps = 1e-5f);
    mlx::core::array operator()(const mlx::core::array& x) const;
    mlx::core::array* weight_ptr() { return &weight_; }
};

class GemmaAttention {
    const GemmaConfiguration& args_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;

public:
    explicit GemmaAttention(const GemmaConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GemmaMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    GemmaMLP(int dimensions, int hidden_dimensions);
    // Uses GELU activation (unlike Llama's SiLU)
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GemmaTransformerBlock {
    GemmaAttention attention_;
    GemmaMLP mlp_;
    GemmaRMSNorm input_layernorm_;
    GemmaRMSNorm post_attention_layernorm_;

public:
    explicit GemmaTransformerBlock(const GemmaConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GemmaModelInner {
    const GemmaConfiguration& args_;
    mlx::core::array embed_tokens_weight_;
    std::vector<GemmaTransformerBlock> layers_;
    GemmaRMSNorm norm_;

public:
    explicit GemmaModelInner(const GemmaConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    // Gemma always uses tied embeddings as lm_head.
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class GemmaModel
    : public LanguageModel<GemmaModel>,
      public KVCacheDimensionProvider<GemmaModel> {

    friend class LanguageModel<GemmaModel>;
    friend class KVCacheDimensionProvider<GemmaModel>;

    GemmaConfiguration config_;
    GemmaModelInner model_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit GemmaModel(const GemmaConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

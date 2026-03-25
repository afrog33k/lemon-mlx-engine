// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx-lm/llm/models/gemma.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Gemma2Configuration {
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
    float attn_logit_softcapping = 50.0f;
    float final_logit_softcapping = 30.0f;
    float query_pre_attn_scalar = 144.0f;
};

void from_json(const nlohmann::json& j, Gemma2Configuration& c);

class Gemma2Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int repeats_;
    float scale_;
    float logit_soft_cap_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    float rope_theta_;
    bool rope_traditional_;

public:
    explicit Gemma2Attention(const Gemma2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma2MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Gemma2MLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma2TransformerBlock {
    Gemma2Attention attention_;
    Gemma2MLP mlp_;
    GemmaRMSNorm input_layernorm_;
    GemmaRMSNorm pre_feedforward_layernorm_;
    GemmaRMSNorm post_feedforward_layernorm_;
    GemmaRMSNorm post_attention_layernorm_;

public:
    explicit Gemma2TransformerBlock(const Gemma2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma2ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Gemma2TransformerBlock> layers_;
    GemmaRMSNorm norm_;
    float hidden_scale_;

public:
    explicit Gemma2ModelInner(const Gemma2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma2Model
    : public LanguageModel<Gemma2Model>,
      public KVCacheDimensionProvider<Gemma2Model> {

    friend class LanguageModel<Gemma2Model>;
    friend class KVCacheDimensionProvider<Gemma2Model>;

    Gemma2Configuration config_;
    Gemma2ModelInner model_;
    float logit_soft_cap_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Gemma2Model(const Gemma2Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

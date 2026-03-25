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

struct Starcoder2Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    float norm_epsilon = 1e-5f;
    int vocab_size = 49152;
    float rope_theta = 100000.0f;
    bool tie_word_embeddings = true;
};

void from_json(const nlohmann::json& j, Starcoder2Configuration& c);

class Starcoder2Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_, wo_bias_;

    float rope_theta_;

public:
    explicit Starcoder2Attention(const Starcoder2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Starcoder2MLP {
    mlx::core::array c_fc_weight_, c_fc_bias_;
    mlx::core::array c_proj_weight_, c_proj_bias_;

public:
    Starcoder2MLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Starcoder2TransformerBlock {
    Starcoder2Attention attention_;
    Starcoder2MLP mlp_;
    mlx::core::array input_layernorm_weight_, input_layernorm_bias_;
    mlx::core::array post_attention_layernorm_weight_, post_attention_layernorm_bias_;
    float norm_eps_;

public:
    explicit Starcoder2TransformerBlock(const Starcoder2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Starcoder2ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Starcoder2TransformerBlock> layers_;
    mlx::core::array norm_weight_, norm_bias_;
    float norm_eps_;

public:
    explicit Starcoder2ModelInner(const Starcoder2Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Starcoder2Model
    : public LanguageModel<Starcoder2Model>,
      public KVCacheDimensionProvider<Starcoder2Model> {

    friend class LanguageModel<Starcoder2Model>;
    friend class KVCacheDimensionProvider<Starcoder2Model>;

    Starcoder2Configuration config_;
    Starcoder2ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Starcoder2Model(const Starcoder2Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

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

struct PhiConfiguration {
    int max_position_embeddings = 2048;
    int vocab_size = 51200;
    int hidden_size = 2560;
    int num_attention_heads = 32;
    int num_hidden_layers = 32;
    int num_key_value_heads = 32;
    float partial_rotary_factor = 0.4f;
    int intermediate_size = 10240;
    float layer_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

void from_json(const nlohmann::json& j, PhiConfiguration& c);

class PhiAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int rope_dim_;

    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array dense_weight_, dense_bias_;

    float rope_theta_;

public:
    explicit PhiAttention(const PhiConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiMLP {
    mlx::core::array fc1_weight_, fc1_bias_;
    mlx::core::array fc2_weight_, fc2_bias_;

public:
    explicit PhiMLP(const PhiConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiDecoderLayer {
    PhiAttention attention_;
    PhiMLP mlp_;
    mlx::core::array input_layernorm_weight_, input_layernorm_bias_;
    float norm_eps_;

public:
    explicit PhiDecoderLayer(const PhiConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<PhiDecoderLayer> layers_;
    mlx::core::array final_layernorm_weight_, final_layernorm_bias_;
    float norm_eps_;

public:
    explicit PhiModelInner(const PhiConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class PhiModel
    : public LanguageModel<PhiModel>,
      public KVCacheDimensionProvider<PhiModel> {

    friend class LanguageModel<PhiModel>;
    friend class KVCacheDimensionProvider<PhiModel>;

    PhiConfiguration config_;
    PhiModelInner model_;
    mlx::core::array lm_head_weight_, lm_head_bias_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit PhiModel(const PhiConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

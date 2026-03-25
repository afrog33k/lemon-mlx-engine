// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of BailingMoe.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct BailingMoeConfiguration {
    int hidden_size;
    int intermediate_size;
    int moe_intermediate_size;
    int num_experts;
    int num_shared_experts;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int num_experts_per_tok;
    int first_k_dense_replace;
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;
    bool norm_topk_prob;
    bool use_bias = false;
    bool use_qkv_bias = false;
    bool use_qk_norm = false;
    bool tie_word_embeddings = false;
    float partial_rotary_factor = 1.0f;
    float routed_scaling_factor = 1.0f;
    std::string score_function = "softmax";
    int n_group = 1;
    int topk_group = 4;
    std::optional<int> moe_shared_expert_intermediate_size;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, BailingMoeConfiguration& c);

class BailingMoeAttention {
    int num_heads_, num_kv_heads_, head_dim_, rope_dim_;
    float scale_;
    mlx::core::array qkv_weight_;
    std::optional<mlx::core::array> qkv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    bool has_qk_norm_;
    float rms_norm_eps_, rope_theta_;
public:
    explicit BailingMoeAttention(const BailingMoeConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
    std::optional<mlx::core::array> gate_bias_, up_bias_, down_bias_;
public:
    BailingMoeMLP(int hidden_size, int inter_size, bool use_bias);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeGate {
    int top_k_, n_group_, topk_group_, num_experts_;
    float routed_scaling_factor_;
    bool norm_topk_prob_;
    mlx::core::array gate_proj_weight_;
    mlx::core::array expert_bias_;
public:
    explicit BailingMoeGate(const BailingMoeConfiguration& config);
    std::pair<mlx::core::array, mlx::core::array> group_select(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeSparseMoeBlock {
    BailingMoeGate gate_;
    SwitchGLU switch_mlp_;
    std::optional<BailingMoeMLP> shared_experts_;
public:
    explicit BailingMoeSparseMoeBlock(const BailingMoeConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeBlock {
    BailingMoeAttention attention_;
    bool use_moe_;
    std::optional<BailingMoeMLP> dense_mlp_;
    std::optional<BailingMoeSparseMoeBlock> moe_mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float norm_eps_;
public:
    BailingMoeBlock(const BailingMoeConfiguration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<BailingMoeBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit BailingMoeModelInner(const BailingMoeConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class BailingMoeModel
    : public LanguageModel<BailingMoeModel>,
      public KVCacheDimensionProvider<BailingMoeModel> {

    friend class LanguageModel<BailingMoeModel>;
    friend class KVCacheDimensionProvider<BailingMoeModel>;

    BailingMoeConfiguration config_;
    BailingMoeModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit BailingMoeModel(const BailingMoeConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

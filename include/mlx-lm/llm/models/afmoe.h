// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of AfMoE.swift
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

struct AfMoEConfiguration {
    int vocab_size = 200192;
    int hidden_size = 2048;
    int intermediate_size = 6144;
    int moe_intermediate_size = 1024;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 4;
    int head_dim = 64;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    bool tie_word_embeddings = false;
    int num_experts = 128;
    int num_experts_per_tok = 8;
    int num_shared_experts = 1;
    int num_dense_layers = 2;
    bool route_norm = true;
    float route_scale = 2.826f;
    std::string score_func = "sigmoid";
    int n_group = 1;
    int topk_group = 1;
    std::vector<std::string> layer_types;
    int sliding_window = 2048;
    bool mup_enabled = true;
};

void from_json(const nlohmann::json& j, AfMoEConfiguration& c);

class AfMoEAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    bool is_local_attention_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    mlx::core::array gate_proj_weight_;  // attention gating
    float rms_norm_eps_, rope_theta_;
    bool has_rope_;
public:
    AfMoEAttention(const AfMoEConfiguration& config, bool is_local_attention);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class AfMoEMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
public:
    AfMoEMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class AfMoEMoE {
    int num_experts_, num_experts_per_tok_, n_group_, topk_group_;
    bool route_norm_;
    float route_scale_;
    std::string score_func_;
    mlx::core::array router_gate_weight_;
    mlx::core::array expert_bias_;
    SwitchGLU experts_;
    std::optional<AfMoEMLP> shared_experts_;
public:
    explicit AfMoEMoE(const AfMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class AfMoEBlock {
    AfMoEAttention self_attn_;
    bool use_moe_;
    std::optional<AfMoEMLP> dense_mlp_;
    std::optional<AfMoEMoE> moe_mlp_;
    bool use_sliding_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array pre_mlp_layernorm_weight_;
    mlx::core::array post_mlp_layernorm_weight_;
    float norm_eps_;
public:
    AfMoEBlock(const AfMoEConfiguration& config, int layer_idx, bool use_sliding);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    bool uses_sliding() const { return use_sliding_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class AfMoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<AfMoEBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
    bool mup_enabled_;
    int hidden_size_;
    int fa_idx_;
    int swa_idx_;  // -1 if none
    int sliding_window_;
public:
    explicit AfMoEModelInner(const AfMoEConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    const std::vector<AfMoEBlock>& get_layers() const { return layers_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class AfMoEModel
    : public LanguageModel<AfMoEModel>,
      public KVCacheDimensionProvider<AfMoEModel> {

    friend class LanguageModel<AfMoEModel>;
    friend class KVCacheDimensionProvider<AfMoEModel>;

    AfMoEConfiguration config_;
    AfMoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;
    std::vector<bool> layer_uses_sliding_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit AfMoEModel(const AfMoEConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

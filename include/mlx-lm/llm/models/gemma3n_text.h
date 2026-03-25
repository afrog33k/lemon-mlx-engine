// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3nText.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// ── Configuration ──────────────────────────────────────────────────────────

struct Gemma3nTextConfiguration {
    int hidden_size;
    int num_hidden_layers;
    std::vector<int> intermediate_size; // per-layer or single value repeated
    int num_attention_heads;
    int head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int num_kv_shared_layers;
    std::optional<float> query_pre_attn_scalar;
    int vocab_size_per_layer_input;
    int hidden_size_per_layer_input;
    int sliding_window;
    int max_position_embeddings;
    float rope_local_base_freq;
    float rope_theta;
    float final_logit_softcapping;
    std::optional<std::vector<std::string>> layer_types;
    std::optional<std::vector<float>> activation_sparsity_pattern;
    int altup_num_inputs;
    std::optional<float> altup_coef_clip;
    bool altup_correct_scale;
    int altup_active_idx;
    int laurel_rank;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    std::optional<int> sliding_window_pattern;
};

void from_json(const nlohmann::json& j, Gemma3nTextConfiguration& c);

// ── Helper: resolve layer types ────────────────────────────────────────────
// Returns the configured layer_types, or a default vector of "global_attention".
inline std::vector<std::string> gemma3n_resolve_layer_types(
    const Gemma3nTextConfiguration& config)
{
    if (config.layer_types.has_value()) return config.layer_types.value();
    return std::vector<std::string>(
        static_cast<size_t>(config.num_hidden_layers), "global_attention");
}

// ── Laurel Block ───────────────────────────────────────────────────────────
// Low-rank residual branch: left → right → norm → add

class Gemma3nTextLaurelBlock {
    mlx::core::array linear_left_weight_;   // [hidden_size, laurel_rank]
    mlx::core::array linear_right_weight_;  // [laurel_rank, hidden_size]
    mlx::core::array post_laurel_norm_weight_; // [hidden_size]
    float rms_norm_eps_;

public:
    explicit Gemma3nTextLaurelBlock(const Gemma3nTextConfiguration& config);

    mlx::core::array operator()(const mlx::core::array& x);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Attention ──────────────────────────────────────────────────────────────
// Q/K/V/O projections with Q-norm, K-norm, V-norm (RMSNoScale), RoPE.
// Supports KV sharing for shared-KV layers.

class Gemma3nAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    bool is_sliding_;
    bool is_kv_shared_layer_;
    float rms_norm_eps_;
    float rope_base_freq_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    // v_norm uses RMSNoScale (no learned weight — functional only)
    float v_norm_eps_;

public:
    Gemma3nAttention(const Gemma3nTextConfiguration& config, int layer_idx);

    bool is_sliding() const { return is_sliding_; }
    bool is_kv_shared_layer() const { return is_kv_shared_layer_; }

    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── MLP ────────────────────────────────────────────────────────────────────
// Gated MLP with GELU activation and optional activation sparsity (geluTopK).

class Gemma3nMLP {
    mlx::core::array gate_weight_, up_weight_, down_weight_;
    int hidden_size_;
    int intermediate_size_;
    float activation_sparsity_;
    std::optional<mlx::core::array> std_multiplier_; // for geluTopK

public:
    Gemma3nMLP(const Gemma3nTextConfiguration& config, int layer_idx);

    mlx::core::array operator()(const mlx::core::array& x);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── AltUp ──────────────────────────────────────────────────────────────────
// Alternating updates: predict, correct, and scale multiple input streams.

class Gemma3nAltUp {
    int altup_num_inputs_;
    int altup_active_idx_;
    int hidden_size_;
    float rms_norm_eps_;
    std::optional<float> altup_coef_clip_;
    bool altup_correct_scale_;

    mlx::core::array correct_output_scale_weight_;   // [hidden_size]
    mlx::core::array correction_coefs_weight_;        // [altup_num_inputs, altup_num_inputs]
    mlx::core::array prediction_coefs_weight_;        // [altup_num_inputs * altup_num_inputs, altup_num_inputs]
    mlx::core::array modality_router_weight_;         // [altup_num_inputs, hidden_size]
    mlx::core::array router_norm_weight_;             // [hidden_size]
    mlx::core::array router_input_scale_;             // scalar

public:
    explicit Gemma3nAltUp(const Gemma3nTextConfiguration& config);

    // Predict multi-stream outputs from current state.
    mlx::core::array predict(const mlx::core::array& x);

    // Correct predictions using activated representation.
    mlx::core::array correct(const mlx::core::array& predictions,
                              const mlx::core::array& activated);

    // Full forward: returns (corrected_all, corrected_active).
    std::pair<mlx::core::array, mlx::core::array>
    operator()(const mlx::core::array& x, const mlx::core::array& activated);

    // Access the scale weight for external use.
    const mlx::core::array& correct_output_scale() const { return correct_output_scale_weight_; }

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Decoder Layer ──────────────────────────────────────────────────────────

class Gemma3nDecoderLayer {
    Gemma3nAttention self_attn_;
    Gemma3nMLP mlp_;
    Gemma3nAltUp altup_;
    Gemma3nTextLaurelBlock laurel_;

    // 5 norm weights
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array pre_feedforward_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    mlx::core::array post_per_layer_input_norm_weight_;

    // Per-layer input gate / projection
    mlx::core::array per_layer_input_gate_weight_;   // [hidden_size_per_layer_input, hidden_size]
    mlx::core::array per_layer_projection_weight_;   // [hidden_size, hidden_size_per_layer_input]

    float rms_norm_eps_;
    int hidden_size_;
    bool is_sliding_;
    int sliding_window_;
    int altup_active_idx_;
    bool altup_correct_scale_;

public:
    Gemma3nDecoderLayer(const Gemma3nTextConfiguration& config, int layer_idx);

    bool is_sliding() const { return is_sliding_; }

    // Forward pass. per_layer_input is required.
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache,
                                 const mlx::core::array& per_layer_input);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Inner Model ────────────────────────────────────────────────────────────

class Gemma3nModelInner {
    Gemma3nTextConfiguration config_;
    mlx::core::array embed_tokens_weight_;           // [vocab_size, hidden_size]
    mlx::core::array embed_tokens_per_layer_weight_; // [vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]
    std::vector<Gemma3nDecoderLayer> layers_;

    // AltUp projections: (altup_num_inputs - 1) each
    std::vector<mlx::core::array> altup_projections_weights_;       // each [hidden_size, hidden_size]
    std::vector<mlx::core::array> altup_unembed_projections_weights_; // each [hidden_size, hidden_size]

    mlx::core::array per_layer_model_projection_weight_; // [num_hidden_layers * hidden_size_per_layer_input, hidden_size]
    mlx::core::array per_layer_projection_norm_weight_;  // [hidden_size_per_layer_input]
    mlx::core::array norm_weight_;                       // [hidden_size]

    float rms_norm_eps_;
    int hidden_size_;
    int vocab_size_;
    int vocab_size_per_layer_input_;
    int hidden_size_per_layer_input_;
    int num_hidden_layers_;
    int altup_num_inputs_;
    int altup_active_idx_;
    std::optional<float> final_logit_softcapping_;

    // Precomputed layer/cache mapping
    std::vector<int> layer_idx_to_cache_idx_;
    int first_kv_shared_layer_idx_;
    int first_sliding_idx_;
    int first_full_idx_;

    // Precomputed embedding scales
    float embed_tokens_scale_;
    float embed_tokens_per_layer_scale_;

    // Helper methods
    mlx::core::array get_per_layer_inputs(const mlx::core::array& input_ids);
    mlx::core::array project_per_layer_inputs(const mlx::core::array& inputs_embeds,
                                               const mlx::core::array& per_layer_inputs);

public:
    explicit Gemma3nModelInner(const Gemma3nTextConfiguration& config);

    // Forward with inputs (token ids).
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 std::vector<KVCache>* cache = nullptr);

    // Forward with inputs_embeds.
    mlx::core::array forward_embeds(const mlx::core::array& inputs_embeds,
                                     const mlx::core::array& per_layer_inputs,
                                     std::vector<KVCache>* cache = nullptr);

    // Create caches for non-shared layers (sliding + full).
    std::vector<KVCache> new_cache() const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-level Model ────────────────────────────────────────────────────────

class Gemma3nTextModel
    : public LanguageModel<Gemma3nTextModel> {

    friend class LanguageModel<Gemma3nTextModel>;

    Gemma3nTextConfiguration config_;
    Gemma3nModelInner language_model_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Gemma3nTextModel(const Gemma3nTextConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

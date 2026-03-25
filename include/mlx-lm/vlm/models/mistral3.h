// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mistral3
// Mistral3 reuses Pixtral's vision model
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/vlm/vlm_model.h>
#include <mlx-lm/vlm/models/pixtral.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// ── Configuration ──────────────────────────────────────────────────────

// Mistral3 reuses Pixtral's vision configuration
using Mistral3VisionConfiguration = PixtralVisionConfiguration;

struct Mistral3VLMTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int head_dim = 0;  // 0 means computed as hidden_size / num_attention_heads
    int max_position_embeddings = 0;
    int num_key_value_heads = 0;
    float rope_theta = 1000000000.0f;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_parameters;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;
    std::vector<std::string> layer_types;  // "full_attention" or "sliding_attention"
    int sliding_window = 0;  // 0 means not set
    bool use_qk_norm = false;

    int effective_head_dim() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
    int effective_num_kv_heads() const {
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
    }
};

struct Mistral3VLMConfiguration {
    Mistral3VLMTextConfiguration text_config;
    Mistral3VisionConfiguration vision_config;
    std::string model_type;
    int ignore_index = -100;
    int image_token_index = 10;
    std::string vision_feature_select_strategy = "full";
    int vision_feature_layer = -1;
    int vocab_size = 32000;
    int spatial_merge_size = 2;
    bool multimodal_projector_bias = false;
};

void from_json(const nlohmann::json& j, Mistral3VLMTextConfiguration& c);
void from_json(const nlohmann::json& j, Mistral3VLMConfiguration& c);

// ── Mistral3 Patch Merger ──────────────────────────────────────────────

// Unfold/im2col operation followed by a linear merging layer
class Mistral3PatchMerger {
    int spatial_merge_size_;
    int patch_size_;
    int hidden_size_;
    mlx::core::array merging_layer_weight_;  // (hidden_size, hidden_size * sms^2)

public:
    Mistral3PatchMerger(int hidden_size, int spatial_merge_size, int patch_size);
    mlx::core::array operator()(const mlx::core::array& image_features,
                                 const std::vector<std::pair<int,int>>& image_sizes);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Mistral3 MultiModal Projector ──────────────────────────────────────

// RMSNorm -> PatchMerger -> linear_1 -> GELU -> linear_2
class Mistral3MultiModalProjector {
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    Mistral3PatchMerger patch_merger_;
    mlx::core::array linear_1_weight_;
    mlx::core::array linear_2_weight_;
    // Optional biases based on config
    std::optional<mlx::core::array> linear_1_bias_;
    std::optional<mlx::core::array> linear_2_bias_;

public:
    explicit Mistral3MultiModalProjector(const Mistral3VLMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::vector<std::pair<int,int>>& image_sizes);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components (Ministral3) ───────────────────────────────────

// Language Attention with Llama4 attention scaling, RoPE from rope_parameters
class Mistral3LanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_;
    float rope_theta_;
    bool rope_traditional_;
    float rope_scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;

public:
    explicit Mistral3LanguageAttention(const Mistral3VLMTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& attention_scale,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP: gate/up/down (no bias), SiLU
class Mistral3LanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Mistral3LanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Transformer Block with attention scaling
class Mistral3TransformerBlock {
    Mistral3LanguageAttention attention_;
    Mistral3LanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
    bool use_sliding_;

public:
    Mistral3TransformerBlock(const Mistral3VLMTextConfiguration& args, bool use_sliding = false);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& attention_scale,
        const AttentionMask& mask,
        KVCache* cache);
    bool is_sliding() const { return use_sliding_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Ministral3 Model Inner: with sliding window attention and Llama4 scaling
class Mistral3LanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Mistral3TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    std::vector<std::string> layer_types_;
    int sliding_window_;
    int fa_index_;          // Index of first full_attention layer
    int swa_index_;         // Index of first sliding_attention layer (-1 if none)
    float llama4_beta_;
    int original_max_pos_;

public:
    explicit Mistral3LanguageModelInner(const Mistral3VLMTextConfiguration& args);
    mlx::core::array operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model wrapper
class Mistral3LanguageModel {
    Mistral3LanguageModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;
    bool tie_word_embeddings_;
    std::vector<std::string> layer_types_;
    int sliding_window_;

public:
    explicit Mistral3LanguageModel(const Mistral3VLMTextConfiguration& args);
    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Mistral3LanguageModelInner& inner() { return model_; }
    std::vector<KVCache> new_cache(const GenerateParameters& params = {}) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level Mistral3 Model ──────────────────────────────────────────

class Mistral3Model
    : public VLMModel<Mistral3Model> {

    friend class LanguageModel<Mistral3Model>;

    Mistral3VLMConfiguration config_;
    PixtralVisionModel vision_tower_;
    Mistral3LanguageModel language_model_;
    Mistral3MultiModalProjector multi_modal_projector_;
    std::vector<int> kv_heads_cache_;
    int vision_feature_layer_;

    // Merge vision features into text embeddings
    mlx::core::array get_input_embeddings(
        const mlx::core::array& input_ids,
        const mlx::core::array* pixel_values,
        const std::vector<std::pair<int,int>>* image_sizes);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

    // Custom new_cache_impl for per-layer sliding vs full attention
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params) const;

public:
    explicit Mistral3Model(const Mistral3VLMConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/pixtral
#pragma once

#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/vlm/vlm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// ── Configuration ──────────────────────────────────────────────────────

struct PixtralVisionConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int intermediate_size;
    int patch_size;
    int image_size;
    int num_channels = 3;
    float rms_norm_eps = 1e-5f;
    int head_dim = 0;  // 0 means computed as hidden_size / num_attention_heads
    float rope_theta = 10000.0f;

    int effective_head_dim() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
};

struct PixtralTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int head_dim = 0;  // 0 means computed as hidden_size / num_attention_heads
    int max_position_embeddings = 0;  // 0 means not set
    int num_key_value_heads = 0;  // 0 means same as num_attention_heads
    float rope_theta = 1000000000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;
    bool use_qk_norm = false;

    int effective_head_dim() const {
        return head_dim > 0 ? head_dim : (hidden_size / num_attention_heads);
    }
    int effective_num_kv_heads() const {
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
    }
};

struct PixtralConfiguration {
    PixtralTextConfiguration text_config;
    PixtralVisionConfiguration vision_config;
    std::string model_type;
    int ignore_index = -100;
    int image_token_index = 10;
    std::string vision_feature_select_strategy = "full";
    int vision_feature_layer = -1;
    int vocab_size = 32000;
};

void from_json(const nlohmann::json& j, PixtralVisionConfiguration& c);
void from_json(const nlohmann::json& j, PixtralTextConfiguration& c);
void from_json(const nlohmann::json& j, PixtralConfiguration& c);

// ── Vision Components ──────────────────────────────────────────────────

// 2D Rotary position embeddings for vision transformer.
// Precomputes inv_freq table of shape (max_patches^2, head_dim).
class PixtralVisionRotaryEmbedding {
    int dim_;
    mlx::core::array inv_freq_;  // (max_patches^2, head_dim)

public:
    explicit PixtralVisionRotaryEmbedding(const PixtralVisionConfiguration& config);

    // Returns (cos, sin) each of shape (num_positions, head_dim)
    // where positions are gathered from inv_freq by position_ids.
    std::pair<mlx::core::array, mlx::core::array>
    operator()(const mlx::core::array& x, const mlx::core::array& position_ids);
};

// Vision Attention: separate Q/K/V/O (no bias), 2D RoPE
class PixtralVisionAttention {
    int num_heads_;
    int head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;

public:
    explicit PixtralVisionAttention(const PixtralVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const std::pair<mlx::core::array, mlx::core::array>& position_embeddings,
        const AttentionMask& mask = AttentionMask{});
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: gate/up/down (no bias), SiLU
class PixtralVisionMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    explicit PixtralVisionMLP(const PixtralVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision EncoderLayer: RMSNorm -> Attention + residual, RMSNorm -> MLP + residual
class PixtralVisionEncoderLayer {
    PixtralVisionAttention attention_;
    PixtralVisionMLP feed_forward_;
    mlx::core::array attention_norm_weight_;
    mlx::core::array ffn_norm_weight_;
    float rms_norm_eps_;

public:
    explicit PixtralVisionEncoderLayer(const PixtralVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const std::pair<mlx::core::array, mlx::core::array>& position_embeddings,
        const AttentionMask& mask = AttentionMask{});
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Encoder: stack of EncoderLayer
class PixtralVisionEncoder {
    std::vector<PixtralVisionEncoderLayer> layers_;

public:
    explicit PixtralVisionEncoder(const PixtralVisionConfiguration& config);
    // Returns (output, optional_hidden_states). If output_hidden_states is true,
    // hidden_states includes the input and each layer output.
    std::pair<mlx::core::array, std::vector<mlx::core::array>>
    operator()(const mlx::core::array& x,
               const std::pair<mlx::core::array, mlx::core::array>& position_embeddings,
               const AttentionMask& mask,
               bool output_hidden_states = false);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    size_t num_layers() const { return layers_.size(); }
};

// Pixtral Vision Model Inner: patch_conv -> ln_pre -> transformer
class PixtralVisionModelInner {
    // patch_conv weight stored as [out_channels, patch_size, patch_size, in_channels]
    mlx::core::array patch_conv_weight_;
    mlx::core::array ln_pre_weight_;
    PixtralVisionEncoder transformer_;
    PixtralVisionRotaryEmbedding rope_;
    PixtralVisionConfiguration config_;

public:
    explicit PixtralVisionModelInner(const PixtralVisionConfiguration& config);
    // Returns (encoded, hidden_states_vec)
    std::pair<mlx::core::array, std::vector<mlx::core::array>>
    operator()(const mlx::core::array& x, bool output_hidden_states = false);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Model Wrapper: contains inner model, handles weight sanitization
class PixtralVisionModel {
    PixtralVisionModelInner vision_model_;
    int num_channels_;
    size_t num_encoder_layers_;

public:
    explicit PixtralVisionModel(const PixtralVisionConfiguration& config);
    // Returns (encoded, embeddings, hidden_states)
    struct VisionOutput {
        mlx::core::array encoded;
        mlx::core::array embeddings;
        std::vector<mlx::core::array> hidden_states;
    };
    VisionOutput operator()(const mlx::core::array& x, bool output_hidden_states = false);
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    size_t num_encoder_layers() const;
};

// ── Language Components ────────────────────────────────────────────────

// Language Attention: Q/K/V/O (no bias), standard RoPE, optional QK norms
class PixtralLanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_;
    float rope_theta_;
    bool rope_traditional_;
    float rope_scale_;
    bool use_qk_norm_;
    float rms_norm_eps_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    // Optional QK norms
    mlx::core::array q_norm_weight_, k_norm_weight_;

public:
    explicit PixtralLanguageAttention(const PixtralTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP: gate/up/down (no bias), SiLU
class PixtralLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    PixtralLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Transformer Block
class PixtralTransformerBlock {
    PixtralLanguageAttention attention_;
    PixtralLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit PixtralTransformerBlock(const PixtralTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model Inner: embed_tokens + layers + norm
class PixtralLanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<PixtralTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit PixtralLanguageModelInner(const PixtralTextConfiguration& args);
    mlx::core::array operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model wrapper with optional lm_head
class PixtralLanguageModel {
    PixtralLanguageModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;
    bool tie_word_embeddings_;

public:
    explicit PixtralLanguageModel(const PixtralTextConfiguration& args);
    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    PixtralLanguageModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Multimodal Projector ───────────────────────────────────────────────

// linear_1 (bias) -> GELU -> linear_2 (bias)
class PixtralMultiModalProjector {
    mlx::core::array linear_1_weight_, linear_1_bias_;
    mlx::core::array linear_2_weight_, linear_2_bias_;

public:
    PixtralMultiModalProjector(int vision_hidden_size, int text_hidden_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level Pixtral Model ───────────────────────────────────────────

class PixtralModel
    : public VLMModel<PixtralModel>,
      public KVCacheDimensionProvider<PixtralModel> {

    friend class LanguageModel<PixtralModel>;
    friend class KVCacheDimensionProvider<PixtralModel>;

    PixtralConfiguration config_;
    PixtralVisionModel vision_tower_;
    PixtralLanguageModel language_model_;
    PixtralMultiModalProjector multi_modal_projector_;
    std::vector<int> kv_heads_cache_;
    int vision_feature_layer_;

    // Merge vision features into text embeddings
    mlx::core::array get_input_embeddings(
        const mlx::core::array& input_ids,
        const mlx::core::array* pixel_values);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit PixtralModel(const PixtralConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

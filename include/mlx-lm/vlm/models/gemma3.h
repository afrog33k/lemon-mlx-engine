// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3.swift — Gemma3 VLM (SigLip vision + Gemma3 language)
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/models/gemma.h>
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

struct Gemma3TextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int sliding_window;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    std::optional<float> final_logit_softcapping;

    int vocab_size = 262208;
    float rms_norm_eps = 1e-6f;
    int num_attention_heads = 8;
    int num_key_value_heads = 4;
    int head_dim = 256;
    float query_pre_attn_scalar = 256.0f;
    float rope_theta = 1000000.0f;
    float rope_local_base_freq = 10000.0f;
    bool rope_traditional = false;
    int mm_tokens_per_image = 256;
    int sliding_window_pattern = 6;
    int max_position_embeddings = 4096;
};

struct Gemma3VisionConfiguration {
    std::string model_type;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int patch_size;
    int image_size;

    int num_channels = 3;
    float layer_norm_eps = 1e-6f;

    int num_positions() const {
        return (image_size / patch_size) * (image_size / patch_size);
    }
};

struct Gemma3Configuration {
    Gemma3TextConfiguration text_config;
    Gemma3VisionConfiguration vision_config;
    std::string model_type;
    int mm_tokens_per_image;
    int vocab_size = -1; // -1 means use text_config.vocab_size
    int pad_token_id = 0;

    int effective_vocab_size() const {
        return (vocab_size > 0) ? vocab_size : text_config.vocab_size;
    }
};

void from_json(const nlohmann::json& j, Gemma3TextConfiguration& c);
void from_json(const nlohmann::json& j, Gemma3VisionConfiguration& c);
void from_json(const nlohmann::json& j, Gemma3Configuration& c);

// ── Vision Components (SigLip) ─────────────────────────────────────────

// Patch embedding via Conv2d + learned positional embedding
class Gemma3VisionEmbeddings {
    mlx::core::array patch_embedding_weight_; // Conv2d kernel [out, kH, kW, in]
    mlx::core::array position_embedding_weight_; // [num_positions, hidden_size]
    int patch_size_, hidden_size_, num_positions_;

public:
    Gemma3VisionEmbeddings(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Multi-head self-attention with separate Q/K/V/O projections (all with bias)
class Gemma3VisionAttention {
    int num_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_, wo_bias_;

public:
    Gemma3VisionAttention(int dims, int num_heads);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: fc1 -> GELU(precise) -> fc2 (both with bias)
class Gemma3VisionMLP {
    mlx::core::array fc1_weight_, fc1_bias_;
    mlx::core::array fc2_weight_, fc2_bias_;

public:
    Gemma3VisionMLP(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Pre-norm encoder layer: LayerNorm -> Attention + residual, LayerNorm -> MLP + residual
class Gemma3VisionEncoderLayer {
    Gemma3VisionAttention attention_;
    Gemma3VisionMLP mlp_;
    mlx::core::array layer_norm1_weight_, layer_norm1_bias_;
    mlx::core::array layer_norm2_weight_, layer_norm2_bias_;
    float eps_;

public:
    explicit Gemma3VisionEncoderLayer(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Stack of encoder layers
class Gemma3VisionEncoder {
    std::vector<Gemma3VisionEncoderLayer> layers_;

public:
    explicit Gemma3VisionEncoder(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// SigLip vision model: embeddings -> encoder -> post_layernorm
class Gemma3SigLipVisionModel {
    Gemma3VisionEmbeddings embeddings_;
    Gemma3VisionEncoder encoder_;
    mlx::core::array post_layernorm_weight_, post_layernorm_bias_;
    float eps_;

public:
    explicit Gemma3SigLipVisionModel(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision model wrapper with weight sanitization (conv format conversion)
class Gemma3VisionModel {
    Gemma3SigLipVisionModel vision_model_;
    int num_channels_;

public:
    explicit Gemma3VisionModel(const Gemma3VisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components (Gemma3-style) ─────────────────────────────────

// Gemma3 attention with Q/K norms, sliding window pattern, and per-layer RoPE base
class Gemma3LanguageAttention {
    int heads_, kv_heads_, head_dim_;
    int layer_idx_;
    float scale_;
    bool is_sliding_;      // true if this layer uses sliding window
    float rope_theta_;     // per-layer: global uses rope_theta, sliding uses rope_local_base_freq
    bool rope_traditional_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;

    // Q/K norms (Gemma RMSNorm with 1+weight trick)
    GemmaRMSNorm q_norm_;
    GemmaRMSNorm k_norm_;

public:
    explicit Gemma3LanguageAttention(const Gemma3TextConfiguration& args, int layer_idx);
    bool is_sliding() const { return is_sliding_; }

    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gemma3 MLP: gate/up/down projections with GELU approximate (fast) activation
class Gemma3LanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Gemma3LanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gemma3 transformer block with 4 RMSNorms (all Gemma-style 1+weight)
class Gemma3TransformerBlock {
    Gemma3LanguageAttention attention_;
    Gemma3LanguageMLP mlp_;
    GemmaRMSNorm input_layernorm_;
    GemmaRMSNorm post_attention_layernorm_;
    GemmaRMSNorm pre_feedforward_layernorm_;
    GemmaRMSNorm post_feedforward_layernorm_;

public:
    explicit Gemma3TransformerBlock(const Gemma3TextConfiguration& args, int layer_idx);
    bool is_sliding() const { return attention_.is_sliding(); }

    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Inner language model: embed + layers + norm, with hidden_scale = sqrt(hidden_size)
class Gemma3LanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Gemma3TransformerBlock> layers_;
    GemmaRMSNorm norm_;
    float hidden_scale_;
    int sliding_window_pattern_;
    int sliding_window_;

public:
    explicit Gemma3LanguageModelInner(const Gemma3TextConfiguration& args);
    mlx::core::array operator()(const std::optional<mlx::core::array>& inputs,
                                 std::vector<KVCache>* cache = nullptr,
                                 const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language model wrapper with separate lm_head and optional softcapping
class Gemma3LanguageModel {
    Gemma3LanguageModelInner model_;
    mlx::core::array lm_head_weight_;
    std::optional<float> final_logit_softcapping_;
    std::vector<int> kv_heads_;
    int sliding_window_;
    int sliding_window_pattern_;
    int num_hidden_layers_;

public:
    explicit Gemma3LanguageModel(const Gemma3TextConfiguration& args);
    LMOutput operator()(const std::optional<mlx::core::array>& inputs,
                        std::vector<KVCache>* cache = nullptr,
                        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Gemma3LanguageModelInner& inner() { return model_; }

    // Create appropriate cache types (simple for global, rotating for sliding)
    std::vector<KVCache> new_cache() const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Multimodal Projector (Gemma3MultiModalProjector) ───────────────────

// Projects vision features to language model dimension via:
//   reshape + avgpool + Gemma RMSNorm + matmul with projection weight
class Gemma3MultiModalProjector {
    mlx::core::array mm_input_projection_weight_; // [vision_hidden, text_hidden]
    GemmaRMSNorm mm_soft_emb_norm_;

    int patches_per_image_;
    int tokens_per_side_;
    int kernel_size_;

public:
    Gemma3MultiModalProjector(const Gemma3Configuration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level Gemma3 Model ─────────────────────────────────────────────

class Gemma3Model
    : public VLMModel<Gemma3Model>,
      public KVCacheDimensionProvider<Gemma3Model> {

    friend class LanguageModel<Gemma3Model>;
    friend class KVCacheDimensionProvider<Gemma3Model>;

    Gemma3Configuration config_;
    Gemma3VisionModel vision_tower_;
    Gemma3LanguageModel language_model_;
    Gemma3MultiModalProjector multi_modal_projector_;
    std::vector<int> kv_heads_cache_;

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

    // Override new_cache_impl to create per-layer cache types
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params) const;

public:
    explicit Gemma3Model(const Gemma3Configuration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of PaliGemma.swift — PaliGemma VLM (SigLip vision + Gemma language)
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
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

struct PaliGemmaTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

struct PaliGemmaVisionConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int patch_size;
    int projection_dim;
    int image_size;
    int num_channels = 3;
    float layer_norm_eps = 1e-6f;

    int num_positions() const {
        return (image_size / patch_size) * (image_size / patch_size);
    }
};

struct PaliGemmaConfiguration {
    PaliGemmaTextConfiguration text_config;
    PaliGemmaVisionConfiguration vision_config;
    std::string model_type;
    int vocab_size;
    int ignore_index = -100;
    int image_token_index;
    int hidden_size;
    int pad_token_id;
};

void from_json(const nlohmann::json& j, PaliGemmaTextConfiguration& c);
void from_json(const nlohmann::json& j, PaliGemmaVisionConfiguration& c);
void from_json(const nlohmann::json& j, PaliGemmaConfiguration& c);

// ── Vision Components (SigLip) ─────────────────────────────────────────

// Patch embedding via Conv2d + learned positional embedding
class PaliGemmaVisionEmbeddings {
    mlx::core::array patch_embedding_weight_; // Conv2d kernel [out, kH, kW, in]
    mlx::core::array position_embedding_weight_; // [num_positions, hidden_size]
    int patch_size_, hidden_size_, num_positions_;

public:
    PaliGemmaVisionEmbeddings(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Multi-head self-attention with separate Q/K/V/O projections (all with bias)
class PaliGemmaVisionAttention {
    int num_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_, wo_bias_;

public:
    PaliGemmaVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: fc1 -> GELU -> fc2 (both with bias)
class PaliGemmaVisionMLP {
    mlx::core::array fc1_weight_, fc1_bias_;
    mlx::core::array fc2_weight_, fc2_bias_;

public:
    PaliGemmaVisionMLP(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Pre-norm encoder layer: LayerNorm -> Attention + residual, LayerNorm -> MLP + residual
class PaliGemmaVisionEncoderLayer {
    PaliGemmaVisionAttention attention_;
    PaliGemmaVisionMLP mlp_;
    mlx::core::array layer_norm1_weight_, layer_norm1_bias_;
    mlx::core::array layer_norm2_weight_, layer_norm2_bias_;
    float eps_;

public:
    explicit PaliGemmaVisionEncoderLayer(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Stack of encoder layers
class PaliGemmaVisionEncoder {
    std::vector<PaliGemmaVisionEncoderLayer> layers_;

public:
    explicit PaliGemmaVisionEncoder(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// SigLip vision model: embeddings -> encoder -> post_layernorm
class PaliGemmaSigLipVisionModel {
    PaliGemmaVisionEmbeddings embeddings_;
    PaliGemmaVisionEncoder encoder_;
    mlx::core::array post_layernorm_weight_, post_layernorm_bias_;
    float eps_;

public:
    explicit PaliGemmaSigLipVisionModel(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision model wrapper with weight sanitization (conv format conversion)
class PaliGemmaVisionModel {
    PaliGemmaSigLipVisionModel vision_model_;
    int num_channels_;

public:
    explicit PaliGemmaVisionModel(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components (Gemma-based) ──────────────────────────────────

// Gemma-style attention: no bias on Q/K/V/O, RoPE applied
class PaliGemmaLanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_, rope_theta_;
    bool rope_traditional_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;

public:
    explicit PaliGemmaLanguageAttention(const PaliGemmaTextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gemma MLP: gate/up/down projections, GELU activation
class PaliGemmaLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    PaliGemmaLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gemma transformer block with RMSNorm (1+weight trick)
class PaliGemmaTransformerBlock {
    PaliGemmaLanguageAttention attention_;
    PaliGemmaLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit PaliGemmaTransformerBlock(const PaliGemmaTextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Inner language model: embed + layers + norm, with hidden_scale = sqrt(hidden_size)
class PaliGemmaLanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<PaliGemmaTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    float hidden_scale_;

public:
    explicit PaliGemmaLanguageModelInner(const PaliGemmaTextConfiguration& args);
    mlx::core::array operator()(const std::optional<mlx::core::array>& inputs,
                                 std::vector<KVCache>* cache = nullptr,
                                 const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language model wrapper (always tied embeddings for Gemma)
class PaliGemmaLanguageModel {
    PaliGemmaLanguageModelInner model_;
    std::vector<int> kv_heads_;

public:
    explicit PaliGemmaLanguageModel(const PaliGemmaTextConfiguration& args);
    LMOutput operator()(const std::optional<mlx::core::array>& inputs,
                        std::vector<KVCache>* cache = nullptr,
                        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    PaliGemmaLanguageModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Multimodal Projector ───────────────────────────────────────────────

// Simple linear projection from vision hidden_size to projection_dim (with bias)
class PaliGemmaMultiModalProjector {
    mlx::core::array weight_, bias_;

public:
    PaliGemmaMultiModalProjector(const PaliGemmaVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level PaliGemma Model ──────────────────────────────────────────

class PaliGemmaModel
    : public VLMModel<PaliGemmaModel>,
      public KVCacheDimensionProvider<PaliGemmaModel> {

    friend class LanguageModel<PaliGemmaModel>;
    friend class KVCacheDimensionProvider<PaliGemmaModel>;

    PaliGemmaConfiguration config_;
    PaliGemmaVisionModel vision_tower_;
    PaliGemmaLanguageModel language_model_;
    PaliGemmaMultiModalProjector multi_modal_projector_;
    std::vector<int> kv_heads_cache_;

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit PaliGemmaModel(const PaliGemmaConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

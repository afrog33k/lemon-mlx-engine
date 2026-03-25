// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_vl
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/vlm/vlm_model.h>
#include <mlx-lm/vlm/qwen_vl_utils.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// ── Configuration ──────────────────────────────────────────────────────

struct Qwen2VLTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps = 1e-6f;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 1000000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = true;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

struct Qwen2VLVisionConfiguration {
    int depth;
    int embed_dim;
    int hidden_size;
    int num_heads;
    int patch_size;
    float mlp_ratio = 4.0f;
    int in_channels = 3;
    float layer_norm_eps = 1e-6f;
    int spatial_patch_size;
    int spatial_merge_size = 2;
    int temporal_patch_size = 2;
};

struct Qwen2VLBaseConfiguration {
    std::string model_type;
    int vocab_size;
    int image_token_id;
    int video_token_id;
    int hidden_size;
};

struct Qwen2VLConfiguration {
    Qwen2VLTextConfiguration text_config;
    Qwen2VLVisionConfiguration vision_config;
    Qwen2VLBaseConfiguration base_config;
};

void from_json(const nlohmann::json& j, Qwen2VLTextConfiguration& c);
void from_json(const nlohmann::json& j, Qwen2VLVisionConfiguration& c);
void from_json(const nlohmann::json& j, Qwen2VLBaseConfiguration& c);
void from_json(const nlohmann::json& j, Qwen2VLConfiguration& c);

struct Qwen2VLProcessorConfiguration {
    std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
    int merge_size = 2;
    int patch_size = 14;
    int temporal_patch_size = 2;
    int min_pixels = 3136;
    int max_pixels = 12845056;
};

void from_json(const nlohmann::json& j, Qwen2VLProcessorConfiguration& c);

// ── Vision Components ──────────────────────────────────────────────────

// Vision PatchMerger: reduces spatial tokens via LayerNorm + MLP
class Qwen2VLPatchMerger {
    int hidden_size_;
    mlx::core::array ln_q_weight_, ln_q_bias_;  // LayerNorm
    mlx::core::array mlp_0_weight_, mlp_0_bias_; // Linear 1
    mlx::core::array mlp_2_weight_, mlp_2_bias_; // Linear 2 (after GELU)
    float eps_;

public:
    Qwen2VLPatchMerger(int dimensions, int context_dimensions, int spatial_merge_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Attention: combined QKV with vision rotary embedding
class Qwen2VLVisionAttention {
    int num_heads_;
    float scale_;
    mlx::core::array qkv_weight_, qkv_bias_; // Combined Q/K/V
    mlx::core::array proj_weight_, proj_bias_;

public:
    Qwen2VLVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const std::vector<THW>& frames,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: GELU activation
class Qwen2VLVisionMLP {
    mlx::core::array fc1_weight_, fc1_bias_;
    mlx::core::array fc2_weight_, fc2_bias_;

public:
    Qwen2VLVisionMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Transformer Block
class Qwen2VLVisionBlock {
    Qwen2VLVisionAttention attention_;
    Qwen2VLVisionMLP mlp_;
    mlx::core::array norm1_weight_, norm1_bias_; // LayerNorm 1
    mlx::core::array norm2_weight_, norm2_bias_; // LayerNorm 2
    float eps_;

public:
    explicit Qwen2VLVisionBlock(const Qwen2VLVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& hidden_states,
        const std::vector<THW>& frames,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Model: PatchEmbed + VisionBlocks + PatchMerger
class Qwen2VLVisionModel {
    // PatchEmbed is implemented as a Conv3d weight
    mlx::core::array patch_embed_proj_weight_; // Conv3d kernel
    int patch_size_, temporal_patch_size_, in_channels_, embed_dim_;

    qwen_vl::VisionRotaryEmbedding rotary_pos_emb_;
    std::vector<Qwen2VLVisionBlock> blocks_;
    Qwen2VLPatchMerger merger_;
    int spatial_merge_size_;

    // Compute position embeddings for vision tokens
    mlx::core::array compute_rotary_pos_emb(const std::vector<THW>& frames);
    // Apply PatchEmbed
    mlx::core::array patch_embed(const mlx::core::array& hidden_states);

public:
    explicit Qwen2VLVisionModel(const Qwen2VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& hidden_states, const std::vector<THW>& frames);

    // Weight sanitization for conv weights (PyTorch -> MLX format)
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components ────────────────────────────────────────────────

// Language Attention with multimodal RoPE
class Qwen2VLLanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_;
    float rope_theta_;
    bool rope_traditional_;
    std::vector<int> mrope_section_; // cumsum of mrope_section * 2

    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_;

public:
    explicit Qwen2VLLanguageAttention(const Qwen2VLTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP (SiLU gate)
class Qwen2VLLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Qwen2VLLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Decoder Layer
class Qwen2VLDecoderLayer {
    Qwen2VLLanguageAttention attention_;
    Qwen2VLLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen2VLDecoderLayer(const Qwen2VLTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model Inner (embed + layers + norm)
class Qwen2VLLanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen2VLDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen2VLLanguageModelInner(const Qwen2VLTextConfiguration& args);
    // Can take either input token IDs or pre-computed embeddings
    mlx::core::array operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model (wrapper with optional lm_head)
class Qwen2VLLanguageModel {
    Qwen2VLLanguageModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

public:
    explicit Qwen2VLLanguageModel(const Qwen2VLTextConfiguration& args);
    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Qwen2VLLanguageModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level Model ────────────────────────────────────────────────────

class Qwen2VLModel
    : public VLMModel<Qwen2VLModel>,
      public KVCacheDimensionProvider<Qwen2VLModel> {

    friend class LanguageModel<Qwen2VLModel>;
    friend class KVCacheDimensionProvider<Qwen2VLModel>;

    Qwen2VLConfiguration config_;
    Qwen2VLVisionModel vision_tower_;
    Qwen2VLLanguageModel language_model_;
    std::vector<int> kv_heads_cache_;

    // Merge vision features into text embeddings
    mlx::core::array input_embeddings(
        const mlx::core::array& input_ids,
        const mlx::core::array* pixel_values,
        const std::vector<THW>* frames);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Qwen2VLModel(const Qwen2VLConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

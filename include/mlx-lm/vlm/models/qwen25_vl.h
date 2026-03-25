// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen2_5_vl
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

struct Qwen25VLTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps = 1e-6f;
    int vocab_size;
    int num_key_value_heads;
    int max_position_embeddings = 128000;
    float rope_theta = 1000000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = true;
    int sliding_window = 32768;
    bool use_sliding_window = false;

    int head_dim() const { return hidden_size / num_attention_heads; }
};

struct Qwen25VLVisionConfiguration {
    int depth;
    int hidden_size;
    int intermediate_size;
    int out_hidden_size;
    int num_heads;
    int patch_size;
    int in_channels = 3;
    float layer_norm_eps = 1e-6f;
    int spatial_patch_size;
    int spatial_merge_size;
    int temporal_patch_size;
    int window_size;
    std::vector<int> fullatt_block_indexes;
    int tokens_per_second = 0;
    bool skip_vision = false;
    std::string hidden_act = "silu";
};

struct Qwen25VLBaseConfiguration {
    std::string model_type;
    int vocab_size;
    int image_token_id;
    int video_token_id;
    int vision_start_token_id = 0;
    int vision_end_token_id = 0;
    int vision_token_id = 0;
    int hidden_size;
    int num_attention_heads;
    int num_hidden_layers;
    int intermediate_size;
    int num_key_value_heads;
    int sliding_window = 32768;
    bool use_sliding_window = false;
    int max_window_layers = 0;
};

struct Qwen25VLConfiguration {
    Qwen25VLTextConfiguration text_config;
    Qwen25VLVisionConfiguration vision_config;
    Qwen25VLBaseConfiguration base_config;
};

void from_json(const nlohmann::json& j, Qwen25VLTextConfiguration& c);
void from_json(const nlohmann::json& j, Qwen25VLVisionConfiguration& c);
void from_json(const nlohmann::json& j, Qwen25VLBaseConfiguration& c);
void from_json(const nlohmann::json& j, Qwen25VLConfiguration& c);

struct Qwen25VLProcessorConfiguration {
    std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
    int merge_size = 2;
    int patch_size = 14;
    int temporal_patch_size = 2;
    int min_pixels = 3136;
    int max_pixels = 12845056;
};

void from_json(const nlohmann::json& j, Qwen25VLProcessorConfiguration& c);

// ── Vision Components ──────────────────────────────────────────────────

// Vision PatchMerger: reduces spatial tokens via RMSNorm + MLP (GELU)
// Key difference from Qwen2VL: uses RMSNorm instead of LayerNorm
class Qwen25VLPatchMerger {
    int hidden_size_;
    mlx::core::array ln_q_weight_;  // RMSNorm (no bias)
    mlx::core::array mlp_0_weight_, mlp_0_bias_; // Linear 1
    mlx::core::array mlp_2_weight_, mlp_2_bias_; // Linear 2 (after GELU)
    float eps_;

public:
    Qwen25VLPatchMerger(int dimensions, int context_dimensions, int spatial_merge_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Attention: combined QKV with vision rotary embedding
// Takes an attention mask for windowed/full attention support
class Qwen25VLVisionAttention {
    int num_heads_;
    float scale_;
    mlx::core::array qkv_weight_, qkv_bias_; // Combined Q/K/V
    mlx::core::array proj_weight_, proj_bias_;

public:
    Qwen25VLVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& attention_mask,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: SiLU gate activation (gate/up/down with bias)
// Key difference from Qwen2VL: uses SiLU instead of GELU, gate/up/down instead of fc1/fc2
class Qwen25VLVisionMLP {
    mlx::core::array gate_weight_, gate_bias_;
    mlx::core::array up_weight_, up_bias_;
    mlx::core::array down_weight_, down_bias_;

public:
    Qwen25VLVisionMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Transformer Block
// Key difference from Qwen2VL: uses RMSNorm instead of LayerNorm, passes attention mask
class Qwen25VLVisionBlock {
    Qwen25VLVisionAttention attention_;
    Qwen25VLVisionMLP mlp_;
    mlx::core::array norm1_weight_; // RMSNorm (no bias)
    mlx::core::array norm2_weight_; // RMSNorm (no bias)
    float eps_;

public:
    explicit Qwen25VLVisionBlock(const Qwen25VLVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& hidden_states,
        const mlx::core::array& attention_mask,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Model: PatchEmbed + VisionBlocks (with windowed attention) + PatchMerger
class Qwen25VLVisionModel {
    // PatchEmbed is implemented as a Conv3d weight
    mlx::core::array patch_embed_proj_weight_; // Conv3d kernel
    int patch_size_, temporal_patch_size_, in_channels_, hidden_size_;

    qwen_vl::VisionRotaryEmbedding rotary_pos_emb_;
    std::vector<Qwen25VLVisionBlock> blocks_;
    Qwen25VLPatchMerger merger_;
    int spatial_merge_size_;
    int spatial_merge_unit_;
    int window_size_;
    std::vector<int> fullatt_block_indexes_;

    // Compute position embeddings for vision tokens
    mlx::core::array compute_rotary_pos_emb(const std::vector<THW>& frames);
    // Apply PatchEmbed
    mlx::core::array patch_embed(const mlx::core::array& hidden_states);
    // Compute window indices and cumulative sequence lengths
    std::pair<mlx::core::array, mlx::core::array> get_window_index(const std::vector<THW>& frames);
    // Build block-diagonal attention mask from cumulative sequence lengths
    mlx::core::array build_attention_mask(int sequence_length, const mlx::core::array& cu_seqlens);

public:
    explicit Qwen25VLVisionModel(const Qwen25VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& hidden_states, const std::vector<THW>& frames);

    // Weight sanitization for conv weights (PyTorch -> MLX format)
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components ────────────────────────────────────────────────

// Language Attention with multimodal RoPE
class Qwen25VLLanguageAttention {
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
    explicit Qwen25VLLanguageAttention(const Qwen25VLTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP (SiLU gate)
class Qwen25VLLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Qwen25VLLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Decoder Layer
class Qwen25VLDecoderLayer {
    Qwen25VLLanguageAttention attention_;
    Qwen25VLLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen25VLDecoderLayer(const Qwen25VLTextConfiguration& args);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model Inner (embed + layers + norm)
class Qwen25VLLanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen25VLDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen25VLLanguageModelInner(const Qwen25VLTextConfiguration& args);
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
class Qwen25VLLanguageModel {
    Qwen25VLLanguageModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

public:
    explicit Qwen25VLLanguageModel(const Qwen25VLTextConfiguration& args);
    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Qwen25VLLanguageModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level Model ────────────────────────────────────────────────────

class Qwen25VLModel
    : public VLMModel<Qwen25VLModel>,
      public KVCacheDimensionProvider<Qwen25VLModel> {

    friend class LanguageModel<Qwen25VLModel>;
    friend class KVCacheDimensionProvider<Qwen25VLModel>;

    Qwen25VLConfiguration config_;
    Qwen25VLVisionModel vision_tower_;
    Qwen25VLLanguageModel language_model_;
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
    explicit Qwen25VLModel(const Qwen25VLConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

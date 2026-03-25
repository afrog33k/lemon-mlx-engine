// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Qwen3VL.swift — Qwen3 Vision-Language Model
#pragma once

#include <mlx-lm/common/attention_utils.h>
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

struct Qwen3VLRoPEScaling {
    std::string type;
    bool mrope_interleaved = false;
    std::vector<int> mrope_section = {24, 20, 20};
};

struct Qwen3VLTextConfiguration {
    std::string model_type;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    int max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    std::string hidden_act = "silu";
    std::optional<Qwen3VLRoPEScaling> rope_scaling;
};

struct Qwen3VLVisionConfiguration {
    std::string model_type;
    int depth;
    int hidden_size;
    int intermediate_size;
    int out_hidden_size;
    int num_heads;
    int patch_size;
    int spatial_merge_size = 2;
    int temporal_patch_size = 2;
    int num_position_embeddings;
    int in_channels = 3;
    std::string hidden_act = "gelu";
    std::vector<int> deepstack_visual_indexes;
};

struct Qwen3VLBaseConfiguration {
    std::string model_type;
    int vocab_size;
    int image_token_id = 151655;
    int video_token_id = 151656;
    int vision_start_token_id = 151652;
    int vision_end_token_id = 151653;
    int vision_token_id = 151654;
};

struct Qwen3VLConfiguration {
    Qwen3VLTextConfiguration text_config;
    Qwen3VLVisionConfiguration vision_config;
    Qwen3VLBaseConfiguration base_config;
};

void from_json(const nlohmann::json& j, Qwen3VLRoPEScaling& c);
void from_json(const nlohmann::json& j, Qwen3VLTextConfiguration& c);
void from_json(const nlohmann::json& j, Qwen3VLVisionConfiguration& c);
void from_json(const nlohmann::json& j, Qwen3VLBaseConfiguration& c);
void from_json(const nlohmann::json& j, Qwen3VLConfiguration& c);

// ── Vision Components ──────────────────────────────────────────────────

// Vision PatchEmbed: Conv3d implemented as reshape + matmul
class Qwen3VLPatchEmbed {
    mlx::core::array proj_weight_; // Conv3d kernel [out, T, H, W, C]
    mlx::core::array proj_bias_;
    int patch_size_, temporal_patch_size_, in_channels_, hidden_size_;

public:
    Qwen3VLPatchEmbed(int patch_size, int temporal_patch_size, int in_channels, int hidden_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision PatchMerger: LayerNorm + Linear + GELU + Linear
// Two modes: main merger (usePostShuffleNorm=false) and deepstack (usePostShuffleNorm=true)
class Qwen3VLPatchMerger {
    int hidden_size_;
    bool use_post_shuffle_norm_;
    mlx::core::array norm_weight_, norm_bias_; // LayerNorm
    mlx::core::array linear_fc1_weight_, linear_fc1_bias_;
    mlx::core::array linear_fc2_weight_, linear_fc2_bias_;
    float eps_;

public:
    Qwen3VLPatchMerger(const Qwen3VLVisionConfiguration& config, bool use_post_shuffle_norm);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Attention: combined QKV with bias, proj without bias
class Qwen3VLVisionAttention {
    int num_heads_;
    int head_dim_;
    float scale_;
    mlx::core::array qkv_weight_, qkv_bias_;
    mlx::core::array proj_weight_;

public:
    Qwen3VLVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& cu_seqlens,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: GELU(fast) activation
class Qwen3VLVisionMLP {
    mlx::core::array linear_fc1_weight_, linear_fc1_bias_;
    mlx::core::array linear_fc2_weight_, linear_fc2_bias_;

public:
    Qwen3VLVisionMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Transformer Block
class Qwen3VLVisionBlock {
    Qwen3VLVisionAttention attention_;
    Qwen3VLVisionMLP mlp_;
    mlx::core::array norm1_weight_, norm1_bias_;
    mlx::core::array norm2_weight_, norm2_bias_;
    float eps_;

public:
    explicit Qwen3VLVisionBlock(const Qwen3VLVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& hidden_states,
        const mlx::core::array& cu_seqlens,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Model: PatchEmbed + PosEmbed + VisionBlocks + PatchMerger + DeepstackMergers
class Qwen3VLVisionModel {
    Qwen3VLPatchEmbed patch_embed_;
    qwen_vl::VisionRotaryEmbedding rotary_pos_emb_;
    mlx::core::array pos_embed_weight_; // Learned positional embeddings [N, D]
    std::vector<Qwen3VLVisionBlock> blocks_;
    Qwen3VLPatchMerger merger_;
    std::vector<Qwen3VLPatchMerger> deepstack_mergers_;
    std::vector<int> deepstack_visual_indexes_;
    int spatial_merge_size_;
    int num_grid_per_side_;
    int hidden_size_;
    int in_channels_;

    // Compute rotary position embeddings for vision tokens
    mlx::core::array compute_rotary_pos_emb(const std::vector<THW>& grids);
    // Compute learned positional embeddings with bilinear interpolation
    mlx::core::array compute_positional_embeddings(const std::vector<THW>& grids);
    // Compute cumulative sequence lengths for block-diagonal attention
    mlx::core::array compute_cu_seqlens(const std::vector<THW>& grids);

public:
    explicit Qwen3VLVisionModel(const Qwen3VLVisionConfiguration& config);

    // Returns (merged_features, deepstack_outputs)
    std::pair<mlx::core::array, std::vector<mlx::core::array>>
    operator()(const mlx::core::array& pixel_values, const std::vector<THW>& grid_thw);

    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components ────────────────────────────────────────────────

// Language RotaryEmbedding with interleaved MRoPE
class Qwen3VLRotaryEmbedding {
    mlx::core::array inv_freq_;
    std::vector<int> mrope_section_; // e.g. [24, 20, 20]

    // Apply interleaved MRoPE frequency selection
    mlx::core::array apply_interleaved_mrope(const mlx::core::array& freqs);

public:
    Qwen3VLRotaryEmbedding(int head_dim, float base, const std::optional<Qwen3VLRoPEScaling>& scaling);

    // Returns (cos, sin) each of shape [B, L, head_dim]
    std::pair<mlx::core::array, mlx::core::array>
    operator()(const mlx::core::array& position_ids, mlx::core::Dtype dtype);
};

// Language Attention with Q/K norms and interleaved MRoPE
class Qwen3VLLanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_;

    mlx::core::array wq_weight_;
    mlx::core::array wk_weight_;
    mlx::core::array wv_weight_;
    mlx::core::array wo_weight_;
    mlx::core::array q_norm_weight_; // RMSNorm on head_dim
    mlx::core::array k_norm_weight_; // RMSNorm on head_dim
    float rms_norm_eps_;

    Qwen3VLRotaryEmbedding rotary_emb_;

public:
    explicit Qwen3VLLanguageAttention(const Qwen3VLTextConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache,
        const std::optional<mlx::core::array>& position_ids);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP (SiLU gate)
class Qwen3VLLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    Qwen3VLLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Decoder Layer
class Qwen3VLDecoderLayer {
    Qwen3VLLanguageAttention attention_;
    Qwen3VLLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3VLDecoderLayer(const Qwen3VLTextConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache,
        const std::optional<mlx::core::array>& position_ids);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model Inner (embed_tokens + layers + norm) with deepstack support
class Qwen3VLLanguageModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen3VLDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3VLLanguageModelInner(const Qwen3VLTextConfiguration& config);

    mlx::core::array operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt,
        const AttentionMask& mask = AttentionMask{},
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& visual_mask = std::nullopt,
        const std::vector<mlx::core::array>* deepstack_embeds = nullptr);

    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language Model with MRoPE position ID computation
class Qwen3VLLanguageModel {
    Qwen3VLLanguageModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;
    Qwen3VLConfiguration config_;

    // Persistent rope deltas between prefill and generation
    std::optional<mlx::core::array> rope_deltas_;

public:
    explicit Qwen3VLLanguageModel(const Qwen3VLConfiguration& config);

    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt,
        const AttentionMask& mask = AttentionMask{},
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& visual_mask = std::nullopt,
        const std::vector<mlx::core::array>* deepstack_embeds = nullptr,
        const mlx::core::array* pixel_values = nullptr,
        const std::vector<THW>* image_grid_thw = nullptr,
        const std::vector<THW>* video_grid_thw = nullptr);

    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Qwen3VLLanguageModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Compute MRoPE position IDs for multimodal input
    static std::pair<mlx::core::array, mlx::core::array> get_rope_index(
        const mlx::core::array& input_ids,
        const std::vector<THW>* image_grid_thw,
        const std::vector<THW>* video_grid_thw,
        int spatial_merge_size,
        int image_token_id,
        int video_token_id,
        int vision_start_token_id,
        const mlx::core::array* attention_mask = nullptr);
};

// ── Top-Level Model ────────────────────────────────────────────────────

class Qwen3VLModel
    : public VLMModel<Qwen3VLModel>,
      public KVCacheDimensionProvider<Qwen3VLModel> {

    friend class LanguageModel<Qwen3VLModel>;
    friend class KVCacheDimensionProvider<Qwen3VLModel>;

    Qwen3VLConfiguration config_;
    Qwen3VLVisionModel vision_tower_;
    Qwen3VLLanguageModel language_model_;
    std::vector<int> kv_heads_cache_;

    // Merge vision features into text embeddings, returning (embeddings, visual_mask)
    std::pair<mlx::core::array, mlx::core::array> merge_input_ids_with_image_features(
        const mlx::core::array& image_features,
        const mlx::core::array& input_embeds,
        const mlx::core::array& input_ids,
        int image_token_index,
        int video_token_index);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Qwen3VLModel(const Qwen3VLConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2VL.swift — LFM2 VL VLM (SigLip vision + LFM2 hybrid attention/conv language)
#pragma once

#include <mlx-lm/common/attention_utils.h>
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

struct LFM2VLTextConfiguration {
    std::string model_type;
    int hidden_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int vocab_size;
    float norm_eps = 1e-5f;
    bool conv_bias = false;
    int conv_l_cache = 3;
    int block_dim = -1;          // defaults to hidden_size if not set
    int block_ff_dim = -1;       // defaults to hidden_size if not set
    int block_multiple_of = 256;
    float block_ffn_dim_multiplier = 1.0f;
    bool block_auto_adjust_ff_dim = true;
    std::vector<int> full_attn_idxs;
    float rope_theta = 1000000.0f;

    int head_dim() const { return hidden_size / num_attention_heads; }

    // Computed adjusted FF dim for MLP
    int adjusted_ff_dim() const {
        int ff = block_ff_dim > 0 ? block_ff_dim : hidden_size;
        if (block_auto_adjust_ff_dim) {
            ff = static_cast<int>(static_cast<float>(2 * ff) / 3.0f);
            ff = static_cast<int>(block_ffn_dim_multiplier * static_cast<float>(ff));
            ff = block_multiple_of * ((ff + block_multiple_of - 1) / block_multiple_of);
        }
        return ff;
    }

    int effective_block_dim() const {
        return block_dim > 0 ? block_dim : hidden_size;
    }

    bool is_attention_layer(int idx) const {
        for (int i : full_attn_idxs) {
            if (i == idx) return true;
        }
        return false;
    }
};

struct LFM2VLVisionConfiguration {
    std::string model_type;
    int hidden_size;
    int intermediate_size;
    int num_hidden_layers;
    int num_attention_heads;
    int num_channels = 3;
    int image_size = 224;
    int patch_size = 16;
    int num_patches = 256;
    float layer_norm_eps = 1e-6f;

    int position_embedding_size() const {
        return static_cast<int>(std::sqrt(static_cast<double>(num_patches)));
    }
};

struct LFM2VLConfiguration {
    LFM2VLTextConfiguration text_config;
    LFM2VLVisionConfiguration vision_config;
    std::string model_type;
    int downsample_factor = 2;
    int image_token_index = 396;
    bool projector_bias = true;
    int projector_hidden_size = 2560;
    bool projector_use_layernorm = true;
    int vision_feature_layer = -1;
};

void from_json(const nlohmann::json& j, LFM2VLTextConfiguration& c);
void from_json(const nlohmann::json& j, LFM2VLVisionConfiguration& c);
void from_json(const nlohmann::json& j, LFM2VLConfiguration& c);

// ── Vision Components (SigLip-style) ───────────────────────────────────

// Vision embeddings: Linear patch_embedding + Embedding for position_embedding
// Uses Linear(num_channels * patchSize * patchSize, embed_dim) NOT Conv2d
class LFM2VLVisionEmbeddings {
    mlx::core::array patch_embedding_weight_;    // [embed_dim, num_channels * patchSize^2]
    mlx::core::array position_embedding_weight_; // [num_patches, embed_dim]
    int embed_dim_, patch_size_, num_patches_, position_embedding_size_, num_channels_;

public:
    explicit LFM2VLVisionEmbeddings(const LFM2VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& pixel_values,
                                 const mlx::core::array& spatial_shapes);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Accessor for dtype resolution
    const mlx::core::array& patch_embedding_weight() const { return patch_embedding_weight_; }
};

// Vision multi-head self-attention: Q/K/V/out_proj all with bias
class LFM2VLVisionAttention {
    int num_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_, wo_bias_;

public:
    LFM2VLVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask = AttentionMask{});
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: fc1 (bias) -> GELU approximate -> fc2 (bias)
class LFM2VLVisionMLP {
    mlx::core::array fc1_weight_, fc1_bias_;
    mlx::core::array fc2_weight_, fc2_bias_;

public:
    explicit LFM2VLVisionMLP(const LFM2VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision encoder layer: LayerNorm -> Attention + residual, LayerNorm -> MLP + residual
class LFM2VLVisionEncoderLayer {
    LFM2VLVisionAttention attention_;
    LFM2VLVisionMLP mlp_;
    mlx::core::array layer_norm1_weight_, layer_norm1_bias_;
    mlx::core::array layer_norm2_weight_, layer_norm2_bias_;
    float eps_;

public:
    explicit LFM2VLVisionEncoderLayer(const LFM2VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask = AttentionMask{});
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision encoder: stack of encoder layers, optionally truncated by visionFeatureLayer
class LFM2VLVisionEncoder {
    std::vector<LFM2VLVisionEncoderLayer> layers_;

public:
    LFM2VLVisionEncoder(const LFM2VLVisionConfiguration& config, int vision_feature_layer = -1);
    // Returns the final hidden state after running through all layers
    mlx::core::array operator()(const mlx::core::array& x, bool output_hidden_states = false,
                                 const AttentionMask& mask = AttentionMask{});
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision model: embeddings -> encoder -> post_layernorm
class LFM2VLVisionModel {
    LFM2VLVisionEmbeddings embeddings_;
    LFM2VLVisionEncoder encoder_;
    mlx::core::array post_layernorm_weight_, post_layernorm_bias_;
    float eps_;

public:
    LFM2VLVisionModel(const LFM2VLVisionConfiguration& config, int vision_feature_layer = -1);
    // Returns last hidden state
    mlx::core::array operator()(const mlx::core::array& x, bool output_hidden_states,
                                 const mlx::core::array& spatial_shapes);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    const LFM2VLVisionEmbeddings& embeddings() const { return embeddings_; }
};

// ── Language Components (LFM2 — hybrid attention/conv) ─────────────────

// Conv state for LFM2 short conv layers. Stored per-layer as a simple optional array.
// This parallels the Swift MambaCache: each conv layer stores its rolling state
// of shape [B, l_cache-1, hidden_size].
struct ConvState {
    std::optional<mlx::core::array> state;
};

// LFM2 attention: Q/K/V/out_proj (no bias), q_layernorm/k_layernorm (RMSNorm), RoPE
class LFM2Attention {
    int heads_, kv_heads_, head_dim_;
    float scale_;
    float rope_theta_;
    float norm_eps_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_layernorm_weight_, k_layernorm_weight_;

public:
    explicit LFM2Attention(const LFM2VLTextConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LFM2 short convolution layer: in_proj -> conv1d -> out_proj with cache state
class LFM2ShortConv {
    int l_cache_, hidden_size_;
    bool bias_;
    // Conv1d weights stored as [hidden_size, 1, kernel_size] (depthwise, MLX format)
    mlx::core::array conv_weight_;
    std::optional<mlx::core::array> conv_bias_;
    mlx::core::array in_proj_weight_;
    std::optional<mlx::core::array> in_proj_bias_;
    mlx::core::array out_proj_weight_;
    std::optional<mlx::core::array> out_proj_bias_;

public:
    LFM2ShortConv(const LFM2VLTextConfiguration& config, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x, ConvState* conv_state);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LFM2 MLP: w1/w2/w3 (no bias), SiLU activation
class LFM2MLP {
    mlx::core::array w1_weight_, w2_weight_, w3_weight_;

public:
    explicit LFM2MLP(const LFM2VLTextConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LFM2 decoder layer: either attention OR conv, plus MLP
class LFM2DecoderLayer {
    bool is_attention_layer_;

    // Attention (only present for attention layers)
    std::optional<LFM2Attention> attention_;
    // Conv (only present for conv layers)
    std::optional<LFM2ShortConv> conv_;

    LFM2MLP feed_forward_;
    mlx::core::array operator_norm_weight_; // RMSNorm
    mlx::core::array ffn_norm_weight_;      // RMSNorm
    float norm_eps_;

public:
    LFM2DecoderLayer(const LFM2VLTextConfiguration& config, int layer_idx);
    // For attention layers, uses kv_cache. For conv layers, uses conv_state.
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* kv_cache,
                                 ConvState* conv_state);
    bool is_attention_layer() const { return is_attention_layer_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LFM2 model inner: embed_tokens + layers + embedding_norm
class LFM2ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<LFM2DecoderLayer> layers_;
    mlx::core::array embedding_norm_weight_; // RMSNorm
    float norm_eps_;
    std::vector<int> full_attn_idxs_;

    // Conv states for conv layers, indexed by layer number.
    // Only conv layer slots are used; attention layer slots remain unused.
    std::vector<ConvState> conv_states_;

public:
    explicit LFM2ModelInner(const LFM2VLTextConfiguration& config);
    mlx::core::array operator()(const std::optional<mlx::core::array>& inputs,
                                 std::vector<KVCache>* cache = nullptr,
                                 const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Reset conv states (called when creating a new cache)
    void reset_conv_states();
};

// LFM2 language model: model wrapper, always uses tied embeddings (no separate lm_head)
class LFM2LanguageModel {
    LFM2ModelInner model_;
    std::vector<int> kv_heads_;

public:
    explicit LFM2LanguageModel(const LFM2VLTextConfiguration& config);
    LMOutput operator()(const std::optional<mlx::core::array>& inputs,
                        std::vector<KVCache>* cache = nullptr,
                        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    LFM2ModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Multi-Modal Projector ──────────────────────────────────────────────

// Pixel unshuffle: spatial downsampling via reshape + transpose
class PixelUnshuffleBlock {
    int factor_;

public:
    explicit PixelUnshuffleBlock(int factor);
    mlx::core::array operator()(const mlx::core::array& x);
};

// Multi-modal projector: optional LayerNorm -> linear_1 -> GELU -> linear_2
class LFM2VLMultiModalProjector {
    bool use_layernorm_;
    mlx::core::array layer_norm_weight_, layer_norm_bias_;
    mlx::core::array linear1_weight_;
    std::optional<mlx::core::array> linear1_bias_;
    mlx::core::array linear2_weight_;
    std::optional<mlx::core::array> linear2_bias_;
    float layer_norm_eps_ = 1e-5f;

public:
    explicit LFM2VLMultiModalProjector(const LFM2VLConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level LFM2VL Model ────────────────────────────────────────────

class LFM2VLModel
    : public VLMModel<LFM2VLModel>,
      public KVCacheDimensionProvider<LFM2VLModel> {

    friend class LanguageModel<LFM2VLModel>;
    friend class KVCacheDimensionProvider<LFM2VLModel>;

    LFM2VLConfiguration config_;
    LFM2VLVisionModel vision_tower_;
    LFM2LanguageModel language_model_;
    LFM2VLMultiModalProjector multi_modal_projector_;
    std::optional<PixelUnshuffleBlock> pixel_unshuffle_;
    std::vector<int> kv_heads_cache_;

    // Get input embeddings, merging vision features if present
    mlx::core::array get_input_embeddings(
        const mlx::core::array& input_ids,
        const mlx::core::array* pixel_values,
        const mlx::core::array* spatial_shapes,
        const mlx::core::array* pixel_attention_mask);

    // Merge image features into text embeddings at image token positions
    mlx::core::array merge_input_ids_with_image_features(
        const mlx::core::array& image_features,
        const mlx::core::array& inputs_embeds,
        const mlx::core::array& input_ids,
        int image_token_index);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit LFM2VLModel(const LFM2VLConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Override new_cache_impl to reset conv states and create KV caches
    // for attention layers (conv layers get dummy KVCacheSimple entries).
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params) const;
};

} // namespace mlx_lm

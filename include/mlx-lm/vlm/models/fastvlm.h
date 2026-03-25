// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of FastVLM.swift — FastVLM (FastViTHD vision + Qwen2 language)
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/vlm/vlm_model.h>
#include <mlx-lm/vlm/models/qwen2_vl.h>  // for Qwen2VLTextConfiguration
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx_lm {

// ── Configuration ──────────────────────────────────────────────────────

struct FastVLMVisionConfiguration {
    float cls_ratio = 1.5f;
    int down_patch_size = 7;
    int down_stride = 2;
    std::vector<bool> downsamples;
    std::vector<int> embed_dims;
    int hidden_size = 768;
    int image_size = 512;
    int intermediate_size = 256;
    std::vector<int> layers;
    float layer_scale_init_value = 1e-5f;
    std::vector<int> mlp_ratios;
    int num_classes = 1000;
    int patch_size = 7;
    std::vector<std::optional<std::vector<int>>> pos_embs_shapes;
    int projection_dim = 512;
    int repmixer_kernel_size = 3;
    std::vector<std::string> token_mixers;
};

struct FastVLMBaseConfiguration {
    std::string model_type;
    int image_token_index = -200;
    int eos_token_id = 0;
    std::string mm_projector_type;
    int mm_hidden_size = 0;
    int tokenizer_model_max_length = 0;
    std::string tokenizer_padding_side;
};

struct FastVLMConfiguration {
    Qwen2VLTextConfiguration text_config;
    FastVLMVisionConfiguration vision_config;
    FastVLMBaseConfiguration base_config;
};

void from_json(const nlohmann::json& j, FastVLMVisionConfiguration& c);
void from_json(const nlohmann::json& j, FastVLMBaseConfiguration& c);
void from_json(const nlohmann::json& j, FastVLMConfiguration& c);

// ── Vision Components (FastViTHD, CNN-based) ────────────────────────────

// Helper: Conv2d layer that stores weight/bias and parameters
class FastVLMConv2d {
    mlx::core::array weight_;  // [out_channels, kH, kW, in_channels/groups]
    mlx::core::array bias_;    // [out_channels]
    bool has_bias_;
    std::pair<int,int> stride_;
    std::pair<int,int> padding_;
    std::pair<int,int> dilation_;
    int groups_;

public:
    FastVLMConv2d(int in_channels, int out_channels,
                  std::pair<int,int> kernel_size,
                  std::pair<int,int> stride = {1,1},
                  std::pair<int,int> padding = {0,0},
                  std::pair<int,int> dilation = {1,1},
                  int groups = 1,
                  bool bias = true);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// BatchNorm in inference mode (no tracking, uses running stats)
class FastVLMBatchNormInference {
    mlx::core::array weight_;       // [C]
    mlx::core::array bias_;         // [C]
    mlx::core::array running_mean_; // [C]
    mlx::core::array running_var_;  // [C]
    float eps_;

public:
    FastVLMBatchNormInference(int num_features, float eps = 1e-5f);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Squeeze-and-Excite block
class FastVLMSEBlock {
    FastVLMConv2d reduce_;
    FastVLMConv2d expand_;

public:
    FastVLMSEBlock(int in_channels, float reduction_ratio = 0.0625f);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// MobileOne block (inference reparameterized)
class FastVLMMobileOneBlock {
    FastVLMConv2d reparam_conv_;
    bool use_se_;
    std::optional<FastVLMSEBlock> se_;  // optional SE block

public:
    FastVLMMobileOneBlock(int in_channels, int out_channels,
                          int kernel_size, int stride = 1,
                          int padding = 0, int dilation = 1,
                          int groups = 1, bool use_se = false);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// RepLKNet large kernel conv block (reparameterized)
class FastVLMReparamLargeKernelConv {
    FastVLMConv2d lkb_reparam_;

public:
    FastVLMReparamLargeKernelConv(int in_channels, int out_channels,
                                   int kernel_size, int stride, int groups);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Convolutional patch embedding: ReparamLargeKernelConv + MobileOneBlock
class FastVLMPatchEmbed {
    FastVLMReparamLargeKernelConv proj0_;
    FastVLMMobileOneBlock proj1_;

public:
    FastVLMPatchEmbed(int patch_size, int stride,
                      int in_channels, int embed_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Conditional positional encoding
class FastVLMRepCPE {
    FastVLMConv2d reparam_conv_;

public:
    FastVLMRepCPE(int in_channels, int embed_dim,
                  std::pair<int,int> spatial_shape = {7,7});
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Conv with BatchNorm (no bias on conv)
class FastVLMConvWithNorm {
    FastVLMConv2d conv_;
    FastVLMBatchNormInference bn_;

public:
    FastVLMConvWithNorm(int in_channels, int out_channels);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Convolutional FFN
class FastVLMConvFFN {
    FastVLMConvWithNorm conv_;
    FastVLMConv2d fc1_;
    FastVLMConv2d fc2_;

public:
    FastVLMConvFFN(int in_channels,
                   int hidden_channels = -1,
                   int out_channels = -1);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Reparameterizable token mixer
class FastVLMRepMixer {
    FastVLMConv2d reparam_conv_;

public:
    FastVLMRepMixer(int dim, int kernel_size = 3);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// RepMixer metaformer block
class FastVLMRepMixerBlock {
    FastVLMRepMixer token_mixer_;
    FastVLMConvFFN convffn_;
    mlx::core::array layer_scale_;  // [1,1,dim]

public:
    FastVLMRepMixerBlock(int dim, int kernel_size = 3, float mlp_ratio = 4.0f);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LayerNorm for channel dimension (input: [B,H,W,C])
class FastVLMLayerNormChannel {
    mlx::core::array weight_;  // [C]
    mlx::core::array bias_;    // [C]
    float eps_;

public:
    FastVLMLayerNormChannel(int num_features, float eps = 1e-5f);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Multi-headed self attention for vision
class FastVLMMHSA {
    int head_dim_;
    int num_heads_;
    float scale_;
    mlx::core::array qkv_weight_;  // [dim*3, dim]
    mlx::core::array qkv_bias_;    // [dim*3]
    mlx::core::array proj_weight_; // [dim, dim]
    mlx::core::array proj_bias_;   // [dim]

public:
    FastVLMMHSA(int dim, int head_dim = 32, bool qkv_bias = false);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Attention metaformer block
class FastVLMAttentionBlock {
    FastVLMLayerNormChannel norm_;
    FastVLMMHSA token_mixer_;
    FastVLMConvFFN convffn_;
    mlx::core::array layer_scale1_;  // [1,1,dim]
    mlx::core::array layer_scale2_;  // [1,1,dim]

public:
    FastVLMAttentionBlock(int dim, float mlp_ratio = 4.0f);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Stage wrappers for std::variant network layers
struct FastVLMRepMixerStage {
    std::vector<FastVLMRepMixerBlock> blocks;

    FastVLMRepMixerStage(int dim, int num_blocks, int kernel_size, float mlp_ratio);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

struct FastVLMAttentionStage {
    std::vector<FastVLMAttentionBlock> blocks;

    FastVLMAttentionStage(int dim, int num_blocks, float mlp_ratio);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Network layer variant: RepCPE, RepMixerStage, AttentionStage, or PatchEmbed
using FastVLMNetworkLayer = std::variant<
    FastVLMRepCPE,
    FastVLMRepMixerStage,
    FastVLMAttentionStage,
    FastVLMPatchEmbed>;

// Convolutional stem: 3 MobileOneBlocks
class FastVLMConvolutionalStem {
    std::vector<FastVLMMobileOneBlock> blocks_;

public:
    explicit FastVLMConvolutionalStem(const FastVLMVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Global 2D pooling with linear projection
class FastVLMGlobalPool2D {
    mlx::core::array proj_;  // [in_dim, out_dim]

public:
    FastVLMGlobalPool2D(int in_dim, int out_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// FastViTHD model
class FastViTHDModel {
    FastVLMConvolutionalStem patch_embed_;
    std::vector<FastVLMNetworkLayer> network_;
    FastVLMMobileOneBlock conv_exp_;
    FastVLMGlobalPool2D head_;

public:
    explicit FastViTHDModel(const FastVLMVisionConfiguration& config);

    // Returns (cls_out, image_features, optional_hidden_states)
    struct Output {
        mlx::core::array cls_out;
        mlx::core::array image_features;
    };
    Output operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision model wrapper
class FastVLMVisionModel {
    FastViTHDModel vision_model_;

public:
    explicit FastVLMVisionModel(const FastVLMVisionConfiguration& config);

    // Returns (cls_out, image_features)
    FastViTHDModel::Output operator()(const mlx::core::array& x);

    // Sanitize vision weights
    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components (Qwen2-based, no multimodal RoPE) ──────────────

// Attention with standard RoPE (not multimodal)
class FastVLMLanguageAttention {
    int heads_, kv_heads_, head_dim_;
    float scale_, rope_theta_;
    bool rope_traditional_;
    mlx::core::array wq_weight_, wq_bias_;
    mlx::core::array wk_weight_, wk_bias_;
    mlx::core::array wv_weight_, wv_bias_;
    mlx::core::array wo_weight_;  // no bias on o_proj

public:
    explicit FastVLMLanguageAttention(const Qwen2VLTextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language MLP (SiLU gating)
class FastVLMLanguageMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;

public:
    FastVLMLanguageMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language decoder layer
class FastVLMDecoderLayer {
    FastVLMLanguageAttention attention_;
    FastVLMLanguageMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit FastVLMDecoderLayer(const Qwen2VLTextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language model inner (embed + layers + norm)
class FastVLMQwen2Model {
    mlx::core::array embed_tokens_weight_;
    std::vector<FastVLMDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit FastVLMQwen2Model(const Qwen2VLTextConfiguration& args);
    mlx::core::array operator()(const std::optional<mlx::core::array>& inputs,
                                 std::vector<KVCache>* cache = nullptr,
                                 const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    mlx::core::array embed_tokens(const mlx::core::array& ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Language model wrapper with optional lm_head
class FastVLMLanguageModel {
    FastVLMQwen2Model model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

public:
    explicit FastVLMLanguageModel(const Qwen2VLTextConfiguration& args);
    LMOutput operator()(const std::optional<mlx::core::array>& inputs,
                        std::vector<KVCache>* cache = nullptr,
                        const std::optional<mlx::core::array>& input_embedding = std::nullopt);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    FastVLMQwen2Model& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Multi-Modal Projector ──────────────────────────────────────────────

// Sequential Linear+GELU layers
class FastVLMMultiModalProjector {
    // Each layer has weight + bias
    std::vector<mlx::core::array> weights_;
    std::vector<mlx::core::array> biases_;
    int depth_;

public:
    FastVLMMultiModalProjector(const FastVLMConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Top-Level FastVLM Model ────────────────────────────────────────────

class FastVLMModel
    : public VLMModel<FastVLMModel>,
      public KVCacheDimensionProvider<FastVLMModel> {

    friend class LanguageModel<FastVLMModel>;
    friend class KVCacheDimensionProvider<FastVLMModel>;

    FastVLMConfiguration config_;
    FastVLMVisionModel vision_tower_;
    FastVLMLanguageModel language_model_;
    FastVLMMultiModalProjector mm_projector_;
    std::vector<int> kv_heads_cache_;

    // Get input embeddings with optional image features
    mlx::core::array get_input_embeddings(
        const mlx::core::array& input_ids,
        const mlx::core::array* pixel_values,
        const mlx::core::array* mask);

    // Prepare inputs for multimodal (splice image features into text embeddings)
    mlx::core::array prepare_inputs_for_multimodal(
        const mlx::core::array& image_features,
        const mlx::core::array& input_ids,
        const mlx::core::array* mask);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit FastVLMModel(const FastVLMConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

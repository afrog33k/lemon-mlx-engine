// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of FastVLM.swift — FastVLM (FastViTHD vision + Qwen2 language)

#include <mlx-lm/vlm/models/fastvlm.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <regex>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, FastVLMVisionConfiguration& c) {
    c.cls_ratio = j.value("cls_ratio", 1.5f);
    c.down_patch_size = j.value("down_patch_size", 7);
    c.down_stride = j.value("down_stride", 2);

    if (j.contains("downsamples")) {
        c.downsamples = j["downsamples"].get<std::vector<bool>>();
    }
    if (j.contains("embed_dims")) {
        c.embed_dims = j["embed_dims"].get<std::vector<int>>();
    }
    c.hidden_size = j.value("hidden_size", 768);
    c.image_size = j.value("image_size", 512);
    c.intermediate_size = j.value("intermediate_size", 256);

    if (j.contains("layers")) {
        c.layers = j["layers"].get<std::vector<int>>();
    }
    c.layer_scale_init_value = j.value("layer_scale_init_value", 1e-5f);

    if (j.contains("mlp_ratios")) {
        c.mlp_ratios = j["mlp_ratios"].get<std::vector<int>>();
    }
    c.num_classes = j.value("num_classes", 1000);
    c.patch_size = j.value("patch_size", 7);

    if (j.contains("pos_embs_shapes")) {
        auto& arr = j["pos_embs_shapes"];
        c.pos_embs_shapes.clear();
        for (auto& item : arr) {
            if (item.is_null()) {
                c.pos_embs_shapes.push_back(std::nullopt);
            } else {
                c.pos_embs_shapes.push_back(item.get<std::vector<int>>());
            }
        }
    }

    c.projection_dim = j.value("projection_dim", 512);
    c.repmixer_kernel_size = j.value("repmixer_kernel_size", 3);

    if (j.contains("token_mixers")) {
        c.token_mixers = j["token_mixers"].get<std::vector<std::string>>();
    }
}

void from_json(const nlohmann::json& j, FastVLMBaseConfiguration& c) {
    c.model_type = j.value("model_type", std::string("fastvlm"));
    c.image_token_index = j.value("image_token_index", -200);
    c.eos_token_id = j.value("eos_token_id", 0);
    c.mm_projector_type = j.value("mm_projector_type", std::string("mlp2x_gelu"));
    c.mm_hidden_size = j.value("mm_hidden_size", 0);
    c.tokenizer_model_max_length = j.value("tokenizer_model_max_length", 0);
    c.tokenizer_padding_side = j.value("tokenizer_padding_side", std::string("right"));
}

void from_json(const nlohmann::json& j, FastVLMConfiguration& c) {
    // vision_config is a sub-dictionary
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<FastVLMVisionConfiguration>();
    }
    // text_config and base_config are overlaid at the top level
    // (Qwen2VLTextConfiguration from_json reads from top level)
    c.text_config = j.get<Qwen2VLTextConfiguration>();
    c.base_config = j.get<FastVLMBaseConfiguration>();
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

static mx::array gelu_act(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// ── Vision Components ──────────────────────────────────────────────────

// -- Conv2d Layer --

FastVLMConv2d::FastVLMConv2d(int in_channels, int out_channels,
                              std::pair<int,int> kernel_size,
                              std::pair<int,int> stride,
                              std::pair<int,int> padding,
                              std::pair<int,int> dilation,
                              int groups, bool bias)
    : weight_(mx::zeros({out_channels, kernel_size.first, kernel_size.second, in_channels / groups})),
      bias_(mx::zeros({out_channels})),
      has_bias_(bias),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      groups_(groups)
{}

mx::array FastVLMConv2d::operator()(const mx::array& x) {
    // x: [N, H, W, C] (NHWC)
    auto out = mx::conv2d(x, weight_,
                           /*stride=*/{stride_.first, stride_.second},
                           /*padding=*/{padding_.first, padding_.second},
                           /*dilation=*/{dilation_.first, dilation_.second},
                           /*groups=*/groups_);
    if (has_bias_) {
        out = mx::add(out, bias_);
    }
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMConv2d::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["weight"] = &weight_;
    if (has_bias_) {
        map["bias"] = &bias_;
    }
    return map;
}

// -- BatchNorm Inference --

FastVLMBatchNormInference::FastVLMBatchNormInference(int num_features, float eps)
    : weight_(mx::ones({num_features})),
      bias_(mx::zeros({num_features})),
      running_mean_(mx::zeros({num_features})),
      running_var_(mx::ones({num_features})),
      eps_(eps)
{}

mx::array FastVLMBatchNormInference::operator()(const mx::array& x) {
    // x: [B, H, W, C] — normalize over C dimension
    // Reshape stats to [1, 1, 1, C] for broadcasting
    auto mean = mx::reshape(running_mean_, {1, 1, 1, -1});
    auto var = mx::reshape(running_var_, {1, 1, 1, -1});
    auto w = mx::reshape(weight_, {1, 1, 1, -1});
    auto b = mx::reshape(bias_, {1, 1, 1, -1});

    auto normalized = mx::divide(mx::subtract(x, mean),
                                  mx::sqrt(mx::add(var, mx::array(eps_))));
    return mx::add(mx::multiply(w, normalized), b);
}

std::unordered_map<std::string, mx::array*> FastVLMBatchNormInference::weight_map() {
    return {
        {"weight", &weight_},
        {"bias", &bias_},
        {"running_mean", &running_mean_},
        {"running_var", &running_var_},
    };
}

// -- SE Block --

FastVLMSEBlock::FastVLMSEBlock(int in_channels, float reduction_ratio)
    : reduce_(in_channels, static_cast<int>(static_cast<float>(in_channels) * reduction_ratio),
              {1,1}, {1,1}, {0,0}, {1,1}, 1, true),
      expand_(static_cast<int>(static_cast<float>(in_channels) * reduction_ratio), in_channels,
              {1,1}, {1,1}, {0,0}, {1,1}, 1, true)
{}

mx::array FastVLMSEBlock::operator()(const mx::array& x) {
    int B = x.shape(0);
    int C = x.shape(3);

    // Global average pooling: [B,H,W,C] -> [B,C] -> [B,1,1,C]
    auto pooled = mx::mean(x, {1, 2});
    pooled = mx::reshape(pooled, {B, 1, 1, C});

    auto out = reduce_(pooled);
    out = mx::maximum(out, mx::array(0.0f));  // ReLU
    out = expand_(out);
    out = mx::sigmoid(out);
    out = mx::reshape(out, {B, 1, 1, C});

    return mx::multiply(x, out);
}

std::unordered_map<std::string, mx::array*> FastVLMSEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : reduce_.weight_map()) map["reduce." + k] = v;
    for (auto& [k, v] : expand_.weight_map()) map["expand." + k] = v;
    return map;
}

// -- MobileOne Block --

FastVLMMobileOneBlock::FastVLMMobileOneBlock(int in_channels, int out_channels,
                                              int kernel_size, int stride,
                                              int padding, int dilation,
                                              int groups, bool use_se)
    : reparam_conv_(in_channels, out_channels,
                    {kernel_size, kernel_size},
                    {stride, stride},
                    {padding, padding},
                    {dilation, dilation},
                    groups, true),
      use_se_(use_se)
{
    if (use_se) {
        se_.emplace(out_channels);
    }
}

mx::array FastVLMMobileOneBlock::operator()(const mx::array& x) {
    auto out = reparam_conv_(x);
    if (use_se_ && se_.has_value()) {
        out = (*se_)(out);
    }
    out = gelu_act(out);
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMMobileOneBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : reparam_conv_.weight_map()) map["reparam_conv." + k] = v;
    if (use_se_ && se_.has_value()) {
        for (auto& [k, v] : se_->weight_map()) map["se." + k] = v;
    }
    return map;
}

// -- Reparam Large Kernel Conv --

FastVLMReparamLargeKernelConv::FastVLMReparamLargeKernelConv(
    int in_channels, int out_channels, int kernel_size, int stride, int groups)
    : lkb_reparam_(in_channels, out_channels,
                   {kernel_size, kernel_size},
                   {stride, stride},
                   {kernel_size / 2, kernel_size / 2},
                   {1, 1},
                   groups, true)
{}

mx::array FastVLMReparamLargeKernelConv::operator()(const mx::array& x) {
    return gelu_act(lkb_reparam_(x));
}

std::unordered_map<std::string, mx::array*> FastVLMReparamLargeKernelConv::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : lkb_reparam_.weight_map()) map["lkb_reparam." + k] = v;
    return map;
}

// -- Patch Embed --

FastVLMPatchEmbed::FastVLMPatchEmbed(int patch_size, int stride,
                                      int in_channels, int embed_dim)
    : proj0_(in_channels, embed_dim, patch_size, stride, in_channels),
      proj1_(embed_dim, embed_dim, 1, 1, 0, 1, 1, false)
{}

mx::array FastVLMPatchEmbed::operator()(const mx::array& x) {
    auto out = proj0_(x);
    out = proj1_(out);
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMPatchEmbed::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : proj0_.weight_map()) map["proj.0." + k] = v;
    for (auto& [k, v] : proj1_.weight_map()) map["proj.1." + k] = v;
    return map;
}

// -- RepCPE --

FastVLMRepCPE::FastVLMRepCPE(int in_channels, int embed_dim,
                              std::pair<int,int> spatial_shape)
    : reparam_conv_(in_channels, embed_dim,
                    spatial_shape,
                    {1, 1},
                    {spatial_shape.first / 2, spatial_shape.second / 2},
                    {1, 1},
                    embed_dim, true)
{}

mx::array FastVLMRepCPE::operator()(const mx::array& x) {
    return reparam_conv_(x);
}

std::unordered_map<std::string, mx::array*> FastVLMRepCPE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : reparam_conv_.weight_map()) map["reparam_conv." + k] = v;
    return map;
}

// -- ConvWithNorm --

FastVLMConvWithNorm::FastVLMConvWithNorm(int in_channels, int out_channels)
    : conv_(in_channels, out_channels,
            {7, 7}, {1, 1}, {3, 3}, {1, 1},
            in_channels, false),
      bn_(out_channels)
{}

mx::array FastVLMConvWithNorm::operator()(const mx::array& x) {
    return bn_(conv_(x));
}

std::unordered_map<std::string, mx::array*> FastVLMConvWithNorm::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : conv_.weight_map()) map["conv." + k] = v;
    for (auto& [k, v] : bn_.weight_map()) map["bn." + k] = v;
    return map;
}

// -- ConvFFN --

FastVLMConvFFN::FastVLMConvFFN(int in_channels, int hidden_channels, int out_channels)
    : conv_(in_channels, (out_channels > 0 ? out_channels : in_channels)),
      fc1_(in_channels, (hidden_channels > 0 ? hidden_channels : in_channels),
           {1, 1}, {1, 1}, {0, 0}, {1, 1}, 1, true),
      fc2_((hidden_channels > 0 ? hidden_channels : in_channels),
           (out_channels > 0 ? out_channels : in_channels),
           {1, 1}, {1, 1}, {0, 0}, {1, 1}, 1, true)
{}

mx::array FastVLMConvFFN::operator()(const mx::array& x) {
    auto out = conv_(x);
    out = fc1_(out);
    out = gelu_act(out);
    out = fc2_(out);
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMConvFFN::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : conv_.weight_map()) map["conv." + k] = v;
    for (auto& [k, v] : fc1_.weight_map()) map["fc1." + k] = v;
    for (auto& [k, v] : fc2_.weight_map()) map["fc2." + k] = v;
    return map;
}

// -- RepMixer --

FastVLMRepMixer::FastVLMRepMixer(int dim, int kernel_size)
    : reparam_conv_(dim, dim,
                    {kernel_size, kernel_size},
                    {1, 1},
                    {kernel_size / 2, kernel_size / 2},
                    {1, 1},
                    dim, true)
{}

mx::array FastVLMRepMixer::operator()(const mx::array& x) {
    return reparam_conv_(x);
}

std::unordered_map<std::string, mx::array*> FastVLMRepMixer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : reparam_conv_.weight_map()) map["reparam_conv." + k] = v;
    return map;
}

// -- RepMixer Block --

FastVLMRepMixerBlock::FastVLMRepMixerBlock(int dim, int kernel_size, float mlp_ratio)
    : token_mixer_(dim, kernel_size),
      convffn_(dim, static_cast<int>(static_cast<float>(dim) * mlp_ratio)),
      layer_scale_(mx::ones({1, 1, dim}))
{}

mx::array FastVLMRepMixerBlock::operator()(const mx::array& x) {
    auto out = token_mixer_(x);
    out = mx::add(out, mx::multiply(layer_scale_, convffn_(out)));
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMRepMixerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : token_mixer_.weight_map()) map["token_mixer." + k] = v;
    for (auto& [k, v] : convffn_.weight_map()) map["convffn." + k] = v;
    map["layer_scale"] = &layer_scale_;
    return map;
}

// -- LayerNorm Channel --

FastVLMLayerNormChannel::FastVLMLayerNormChannel(int num_features, float eps)
    : weight_(mx::ones({num_features})),
      bias_(mx::zeros({num_features})),
      eps_(eps)
{}

mx::array FastVLMLayerNormChannel::operator()(const mx::array& x) {
    // x: [B, H, W, C] — normalize over last axis (C)
    auto u = mx::mean(x, -1, true);
    auto diff = mx::subtract(x, u);
    auto s = mx::mean(mx::multiply(diff, diff), -1, true);
    auto result = mx::divide(diff, mx::sqrt(mx::add(s, mx::array(eps_))));
    return mx::add(mx::multiply(weight_, result), bias_);
}

std::unordered_map<std::string, mx::array*> FastVLMLayerNormChannel::weight_map() {
    return {
        {"weight", &weight_},
        {"bias", &bias_},
    };
}

// -- MHSA (Vision) --

FastVLMMHSA::FastVLMMHSA(int dim, int head_dim, bool qkv_bias)
    : head_dim_(head_dim),
      num_heads_(dim / head_dim),
      scale_(std::pow(static_cast<float>(head_dim), -0.5f)),
      qkv_weight_(mx::zeros({dim * 3, dim})),
      qkv_bias_(mx::zeros({dim * 3})),
      proj_weight_(mx::zeros({dim, dim})),
      proj_bias_(mx::zeros({dim}))
{}

mx::array FastVLMMHSA::operator()(const mx::array& x) {
    // x: [B, H, W, C] (NHWC)
    // Transpose to [B, C, H, W] then flatten spatial
    auto xt = mx::transpose(x, {0, 3, 1, 2});
    int B = xt.shape(0);
    int C = xt.shape(1);
    int H = xt.shape(2);
    int W = xt.shape(3);
    int N = H * W;

    // Flatten spatial: [B, C, H*W] -> [B, N, C]
    auto flat = mx::reshape(xt, {B, C, N});
    flat = mx::transpose(flat, {0, 2, 1});  // [B, N, C]

    // Apply QKV projection
    auto qkv = linear_fwd(flat, qkv_weight_, &qkv_bias_);  // [B, N, 3*C]

    // Reshape to [B, N, 3, num_heads, head_dim] then transpose to [3, B, num_heads, N, head_dim]
    qkv = mx::reshape(qkv, {B, N, 3, num_heads_, head_dim_});
    qkv = mx::transpose(qkv, {2, 0, 3, 1, 4});

    // Split into q, k, v along first axis
    auto q = mx::slice(qkv, {0, 0, 0, 0, 0}, {1, B, num_heads_, N, head_dim_});
    q = mx::squeeze(q, 0);
    auto k = mx::slice(qkv, {1, 0, 0, 0, 0}, {2, B, num_heads_, N, head_dim_});
    k = mx::squeeze(k, 0);
    auto v = mx::slice(qkv, {2, 0, 0, 0, 0}, {3, B, num_heads_, N, head_dim_});
    v = mx::squeeze(v, 0);

    // Scaled dot-product attention (no mask for vision)
    auto attn = mx::fast::scaled_dot_product_attention(q, k, v, scale_);

    // Reshape: [B, num_heads, N, head_dim] -> [B, N, C]
    attn = mx::transpose(attn, {0, 2, 1, 3});
    attn = mx::reshape(attn, {B, N, C});

    // Project
    auto out = linear_fwd(attn, proj_weight_, &proj_bias_);

    // Reshape back to [B, H, W, C]
    out = mx::reshape(out, {B, H, W, C});
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMMHSA::weight_map() {
    return {
        {"qkv.weight", &qkv_weight_},
        {"qkv.bias", &qkv_bias_},
        {"proj.weight", &proj_weight_},
        {"proj.bias", &proj_bias_},
    };
}

// -- Attention Block --

FastVLMAttentionBlock::FastVLMAttentionBlock(int dim, float mlp_ratio)
    : norm_(dim),
      token_mixer_(dim),
      convffn_(dim, static_cast<int>(static_cast<float>(dim) * mlp_ratio)),
      layer_scale1_(mx::ones({1, 1, dim})),
      layer_scale2_(mx::ones({1, 1, dim}))
{}

mx::array FastVLMAttentionBlock::operator()(const mx::array& x) {
    auto out = mx::add(x, mx::multiply(layer_scale1_, token_mixer_(norm_(x))));
    out = mx::add(out, mx::multiply(layer_scale2_, convffn_(out)));
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMAttentionBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : norm_.weight_map()) map["norm." + k] = v;
    for (auto& [k, v] : token_mixer_.weight_map()) map["token_mixer." + k] = v;
    for (auto& [k, v] : convffn_.weight_map()) map["convffn." + k] = v;
    map["layerScale1"] = &layer_scale1_;
    map["layerScale2"] = &layer_scale2_;
    return map;
}

// -- RepMixer Stage --

FastVLMRepMixerStage::FastVLMRepMixerStage(int dim, int num_blocks,
                                            int kernel_size, float mlp_ratio) {
    blocks.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks.emplace_back(dim, kernel_size, mlp_ratio);
    }
}

mx::array FastVLMRepMixerStage::operator()(const mx::array& x) {
    auto out = x;
    for (auto& block : blocks) {
        out = block(out);
    }
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMRepMixerStage::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : blocks[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Attention Stage --

FastVLMAttentionStage::FastVLMAttentionStage(int dim, int num_blocks, float mlp_ratio) {
    blocks.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        blocks.emplace_back(dim, mlp_ratio);
    }
}

mx::array FastVLMAttentionStage::operator()(const mx::array& x) {
    auto out = x;
    for (auto& block : blocks) {
        out = block(out);
    }
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMAttentionStage::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : blocks[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Network building --

static std::vector<FastVLMNetworkLayer> build_fastvit_network(
    const FastVLMVisionConfiguration& config)
{
    std::vector<FastVLMNetworkLayer> network;
    int num_stages = static_cast<int>(config.layers.size());

    for (int i = 0; i < num_stages; ++i) {
        // Add positional embedding if specified
        if (i < static_cast<int>(config.pos_embs_shapes.size()) &&
            config.pos_embs_shapes[i].has_value()) {
            auto& shape = config.pos_embs_shapes[i].value();
            network.emplace_back(FastVLMRepCPE(
                config.embed_dims[i],
                config.embed_dims[i],
                {shape[0], shape[1]}
            ));
        }

        // Add stage
        if (config.token_mixers[i] == "repmixer") {
            network.emplace_back(FastVLMRepMixerStage(
                config.embed_dims[i],
                config.layers[i],
                config.repmixer_kernel_size,
                static_cast<float>(config.mlp_ratios[i])
            ));
        } else if (config.token_mixers[i] == "attention") {
            network.emplace_back(FastVLMAttentionStage(
                config.embed_dims[i],
                config.layers[i],
                static_cast<float>(config.mlp_ratios[i])
            ));
        } else {
            throw std::runtime_error("Unknown token mixer type: " + config.token_mixers[i]);
        }

        // Add downsample / patch embed between stages
        if (i < num_stages - 1) {
            if (config.downsamples[i] ||
                config.embed_dims[i] != config.embed_dims[i + 1]) {
                network.emplace_back(FastVLMPatchEmbed(
                    config.down_patch_size,
                    config.down_stride,
                    config.embed_dims[i],
                    config.embed_dims[i + 1]
                ));
            }
        }
    }

    return network;
}

// -- Convolutional Stem --

FastVLMConvolutionalStem::FastVLMConvolutionalStem(const FastVLMVisionConfiguration& config) {
    int in_ch = 3;
    int out_ch = config.embed_dims[0];

    blocks_.reserve(3);
    // Block 0: 3 -> embed[0], k=3, s=2, p=1, g=1
    blocks_.emplace_back(in_ch, out_ch, 3, 2, 1, 1, 1, false);
    // Block 1: embed[0] -> embed[0], k=3, s=2, p=1, g=embed[0] (depthwise)
    blocks_.emplace_back(out_ch, out_ch, 3, 2, 1, 1, out_ch, false);
    // Block 2: embed[0] -> embed[0], k=1, s=1, p=0, g=1
    blocks_.emplace_back(out_ch, out_ch, 1, 1, 0, 1, 1, false);
}

mx::array FastVLMConvolutionalStem::operator()(const mx::array& x) {
    auto out = x;
    for (auto& block : blocks_) {
        out = block(out);
    }
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMConvolutionalStem::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto prefix = "blocks." + std::to_string(i) + ".";
        for (auto& [k, v] : blocks_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Global Pool 2D --

FastVLMGlobalPool2D::FastVLMGlobalPool2D(int in_dim, int out_dim)
    : proj_(mx::zeros({in_dim, out_dim}))
{}

mx::array FastVLMGlobalPool2D::operator()(const mx::array& x) {
    // x: [B, H, W, C] -> mean over spatial -> [B, C]
    auto pooled = mx::mean(x, {1, 2});
    // [B, C] x [C, out_dim] -> [B, out_dim]
    return mx::matmul(pooled, proj_);
}

std::unordered_map<std::string, mx::array*> FastVLMGlobalPool2D::weight_map() {
    return {{"proj", &proj_}};
}

// -- FastViTHD Model --

static int compute_conv_exp_out(const FastVLMVisionConfiguration& config) {
    return static_cast<int>(static_cast<float>(config.embed_dims.back()) * config.cls_ratio);
}

FastViTHDModel::FastViTHDModel(const FastVLMVisionConfiguration& config)
    : patch_embed_(config),
      network_(build_fastvit_network(config)),
      conv_exp_(config.embed_dims.back(),
                compute_conv_exp_out(config),
                3, 1, 1, 1,
                config.embed_dims.back(),
                true),  // useSE = true
      head_(compute_conv_exp_out(config), config.projection_dim)
{}

FastViTHDModel::Output FastViTHDModel::operator()(const mx::array& x) {
    auto out = patch_embed_(x);

    for (auto& layer : network_) {
        out = std::visit([&out](auto& l) -> mx::array {
            return l(out);
        }, layer);
    }

    out = conv_exp_(out);
    auto cls_out = head_(out);

    return {cls_out, out};
}

std::unordered_map<std::string, mx::array*> FastViTHDModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : patch_embed_.weight_map()) map["patch_embed." + k] = v;

    for (size_t i = 0; i < network_.size(); ++i) {
        auto prefix = "network." + std::to_string(i) + ".";
        std::visit([&map, &prefix](auto& layer) {
            for (auto& [k, v] : layer.weight_map()) map[prefix + k] = v;
        }, network_[i]);
    }

    for (auto& [k, v] : conv_exp_.weight_map()) map["conv_exp." + k] = v;
    for (auto& [k, v] : head_.weight_map()) map["head." + k] = v;

    return map;
}

// -- Vision Model --

FastVLMVisionModel::FastVLMVisionModel(const FastVLMVisionConfiguration& config)
    : vision_model_(config)
{}

FastViTHDModel::Output FastVLMVisionModel::operator()(const mx::array& x) {
    return vision_model_(x);
}

std::unordered_map<std::string, mx::array> FastVLMVisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        std::string key = k;

        // Transform layer_scale_ prefix to layerScale
        // e.g., "layer_scale_1" -> "layerScale1", "layer_scale_2" -> "layerScale2"
        {
            std::string::size_type pos = 0;
            while ((pos = key.find("layer_scale_", pos)) != std::string::npos) {
                key.replace(pos, 12, "layerScale");
                pos += 10;  // length of "layerScale"
            }
        }

        // Transform network weight names:
        // vision_model.network.X.Y.Z -> vision_model.network.X.layers.Y.Z
        // Only when Y is a digit (i.e., a block index within a stage)
        {
            std::regex network_regex(
                R"((.+)\.vision_model\.network\.(\d+)\.(\d+)\.(.+))");
            std::smatch match;
            if (std::regex_match(key, match, network_regex)) {
                key = match[1].str() + ".vision_model.network." +
                      match[2].str() + ".layers." +
                      match[3].str() + "." + match[4].str();
            }
        }

        sanitized.insert_or_assign(key, v);
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> FastVLMVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_model_.weight_map()) map["vision_model." + k] = v;
    return map;
}

// ── Language Components (Qwen2-based, standard RoPE) ───────────────────

// -- Language Attention --

FastVLMLanguageAttention::FastVLMLanguageAttention(const Qwen2VLTextConfiguration& args)
    : heads_(args.num_attention_heads),
      kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim()),
      scale_(std::pow(static_cast<float>(args.head_dim()), -0.5f)),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim(), args.hidden_size})),
      wq_bias_(mx::zeros({args.num_attention_heads * args.head_dim()})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim(), args.hidden_size})),
      wk_bias_(mx::zeros({args.num_key_value_heads * args.head_dim()})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim(), args.hidden_size})),
      wv_bias_(mx::zeros({args.num_key_value_heads * args.head_dim()})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim()}))
{}

mx::array FastVLMLanguageAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys    = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values  = linear_fwd(x, wv_weight_, &wv_bias_);

    // Reshape: [B, L, H*D] -> [B, H, L, D]
    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE (standard, not multimodal)
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys    = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> FastVLMLanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

FastVLMLanguageMLP::FastVLMLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array FastVLMLanguageMLP::operator()(const mx::array& x) {
    // down(swiglu(gate(x), up(x)))
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> FastVLMLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Decoder Layer --

FastVLMDecoderLayer::FastVLMDecoderLayer(const Qwen2VLTextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array FastVLMDecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> FastVLMDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Qwen2 Model Inner --

FastVLMQwen2Model::FastVLMQwen2Model(const Qwen2VLTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        layers_.emplace_back(args);
    }
}

mx::array FastVLMQwen2Model::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    mx::array h = [&]() -> mx::array {
        if (input_embedding.has_value()) {
            return input_embedding.value();
        } else if (inputs.has_value()) {
            return mx::take(embed_tokens_weight_, inputs.value(), 0);
        } else {
            throw std::runtime_error("Either inputs or input_embedding must be provided");
        }
    }();

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array FastVLMQwen2Model::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array FastVLMQwen2Model::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> FastVLMQwen2Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Language Model --

FastVLMLanguageModel::FastVLMLanguageModel(const Qwen2VLTextConfiguration& args)
    : model_(args)
{
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

LMOutput FastVLMLanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);
    if (lm_head_weight_.has_value()) {
        out = linear_fwd(out, lm_head_weight_.value());
    } else {
        out = model_.embed_as_linear(out);
    }
    return LMOutput(out);
}

std::unordered_map<std::string, mx::array*> FastVLMLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) {
        map["lm_head.weight"] = &lm_head_weight_.value();
    }
    return map;
}

// ── Multi-Modal Projector ──────────────────────────────────────────────

FastVLMMultiModalProjector::FastVLMMultiModalProjector(const FastVLMConfiguration& config)
    : depth_(1)
{
    int hidden_size = config.text_config.hidden_size;
    int mm_hidden = config.base_config.mm_hidden_size;

    // Parse mm_projector_type to get depth
    std::regex mlp_regex(R"(^mlp(\d+)x_gelu$)");
    std::smatch match;
    if (std::regex_match(config.base_config.mm_projector_type, match, mlp_regex)) {
        depth_ = std::stoi(match[1].str());
    }

    // First layer: mm_hidden_size -> hidden_size
    weights_.push_back(mx::zeros({hidden_size, mm_hidden}));
    biases_.push_back(mx::zeros({hidden_size}));

    // Additional layers: hidden_size -> hidden_size (with GELU between)
    for (int i = 1; i < depth_; ++i) {
        weights_.push_back(mx::zeros({hidden_size, hidden_size}));
        biases_.push_back(mx::zeros({hidden_size}));
    }
}

mx::array FastVLMMultiModalProjector::operator()(const mx::array& x) {
    auto out = linear_fwd(x, weights_[0], &biases_[0]);
    for (int i = 1; i < depth_; ++i) {
        out = gelu_act(out);
        out = linear_fwd(out, weights_[i], &biases_[i]);
    }
    return out;
}

std::unordered_map<std::string, mx::array*> FastVLMMultiModalProjector::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (int i = 0; i < depth_; ++i) {
        auto prefix = "layers." + std::to_string(i * 2) + ".";  // stride by 2 to account for GELU layers
        if (i == 0) {
            prefix = "layers.0.";
        } else {
            // Layer index accounts for GELU layers in between:
            // layers.0 = Linear, layers.1 = GELU, layers.2 = Linear, layers.3 = GELU, etc.
            prefix = "layers." + std::to_string(i * 2) + ".";
        }
        map[prefix + "weight"] = &weights_[i];
        map[prefix + "bias"] = &biases_[i];
    }
    return map;
}

// ── Top-Level FastVLM Model ────────────────────────────────────────────

FastVLMModel::FastVLMModel(const FastVLMConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config),
      mm_projector_(config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array FastVLMModel::get_input_embeddings(
    const mx::array& input_ids,
    const mx::array* pixel_values,
    const mx::array* mask)
{
    if (!pixel_values) {
        return language_model_.inner().embed_tokens(input_ids);
    }

    // Vision: input pixels are [B, C, H, W], transpose to [B, H, W, C] for NHWC
    auto pixels_nhwc = mx::transpose(*pixel_values, {0, 2, 3, 1});
    auto [cls_out, image_features] = vision_tower_(pixels_nhwc);

    // image_features: [B, H, W, C] -> [B, H*W, C]
    int B = image_features.shape(0);
    int H = image_features.shape(1);
    int W = image_features.shape(2);
    int C = image_features.shape(3);
    auto flat_features = mx::reshape(image_features, {B, H * W, C});

    // Project through mm_projector
    auto mm_inputs = mm_projector_(flat_features);

    // Prepare multimodal inputs
    return prepare_inputs_for_multimodal(mm_inputs, input_ids, mask);
}

mx::array FastVLMModel::prepare_inputs_for_multimodal(
    const mx::array& image_features,
    const mx::array& input_ids,
    const mx::array* mask)
{
    // This assumes batch_size == 1 and a single image
    mx::array ids = input_ids;

    if (mask) {
        // Remove padding: find start and end from mask
        // mask: [1, L], argmax gives first non-zero
        auto mask_1d = mx::reshape(*mask, {-1});
        auto ids_1d = mx::reshape(input_ids, {-1});

        // start = argmax(mask_1d), end = start + sum(mask_1d)
        auto start_arr = mx::argmax(mask_1d);
        auto sum_arr = mx::sum(mask_1d);

        // We need to evaluate to get integer values for slicing
        mx::eval({start_arr, sum_arr});
        int start = start_arr.item<int>();
        int len = sum_arr.item<int>();

        ids = mx::slice(ids_1d, {start}, {start + len});
    } else {
        ids = mx::reshape(ids, {-1});
    }

    // Find the image_token_index position
    int image_token_idx = config_.base_config.image_token_index;

    // We need to evaluate ids to get the token values
    mx::eval({ids});
    int L = ids.shape(0);

    // Find position of image token
    int image_pos = 0;
    for (int i = 0; i < L; ++i) {
        auto val = mx::slice(ids, {i}, {i + 1});
        mx::eval({val});
        if (val.item<int>() == image_token_idx) {
            image_pos = i;
            break;
        }
    }

    // Build token array without the image token
    // Remove all occurrences of image_token_idx and embed
    std::vector<int> token_vec;
    token_vec.reserve(L);
    for (int i = 0; i < L; ++i) {
        auto val = mx::slice(ids, {i}, {i + 1});
        mx::eval({val});
        int tok = val.item<int>();
        if (tok != image_token_idx) {
            token_vec.push_back(tok);
        }
    }

    auto tokens = mx::array(token_vec.data(), {static_cast<int>(token_vec.size())}, mx::int32);
    auto token_embeddings = language_model_.inner().embed_tokens(tokens);

    // Split at image_pos
    auto before = mx::slice(token_embeddings, {0, 0}, {image_pos, token_embeddings.shape(1)});
    auto after = mx::slice(token_embeddings, {image_pos, 0},
                           {token_embeddings.shape(0), token_embeddings.shape(1)});

    // image_features: [B, N, D] — take first batch
    auto img_feats = mx::slice(image_features, {0, 0, 0},
                                {1, image_features.shape(1), image_features.shape(2)});
    img_feats = mx::squeeze(img_feats, 0);  // [N, D]

    // Concatenate: [before, image_features, after]
    auto embeddings = mx::concatenate({before, img_feats, after}, 0);

    // Add batch dimension
    return mx::expand_dims(embeddings, 0);
}

PrepareResult FastVLMModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    const mx::array* pixels_ptr = nullptr;
    if (input.image.has_value()) {
        pixels_ptr = &input.image->pixels;
    }

    const mx::array* mask_ptr = nullptr;
    if (input.text.mask.has_value()) {
        mask_ptr = &input.text.mask.value();
    }

    auto embeddings = get_input_embeddings(input_ids, pixels_ptr, mask_ptr);

    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, embeddings);
    return PrepareResult::logits(std::move(result));
}

LMOutput FastVLMModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array FastVLMModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
FastVLMModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Step 1: Replace "mm_projector" with "mm_projector.layers" in key names
    std::unordered_map<std::string, mx::array> sanitized;
    for (auto& [k, v] : weights) {
        std::string key = k;

        // Skip rotary embedding keys
        if (key.find("rotary_emb.inv_freq") != std::string::npos) {
            continue;
        }

        // mm_projector -> mm_projector.layers
        {
            std::string::size_type pos = key.find("mm_projector.");
            if (pos != std::string::npos) {
                // Only transform if not already "mm_projector.layers"
                if (key.find("mm_projector.layers") == std::string::npos) {
                    key.replace(pos, 13, "mm_projector.layers.");
                }
            }
        }

        sanitized.insert_or_assign(key, v);
    }

    // Step 2: Apply vision model sanitization
    return vision_tower_.sanitize(std::move(sanitized));
}

void FastVLMModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> FastVLMModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // Vision tower: prefix "vision_tower."
    for (auto& [k, v] : vision_tower_.weight_map())
        map["vision_tower." + k] = v;

    // Language model: prefix "language_model."
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;

    // Multi-modal projector: prefix "mm_projector."
    for (auto& [k, v] : mm_projector_.weight_map())
        map["mm_projector." + k] = v;

    return map;
}

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of PaliGemma.swift — PaliGemma VLM (SigLip vision + Gemma language)

#include <mlx-lm/vlm/models/paligemma.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, PaliGemmaTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("gemma"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
}

void from_json(const nlohmann::json& j, PaliGemmaVisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("siglip_vision_model"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.projection_dim = j.at("projection_dim").get<int>();
    c.image_size = j.at("image_size").get<int>();
    c.num_channels = j.value("num_channels", 3);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
}

void from_json(const nlohmann::json& j, PaliGemmaConfiguration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<PaliGemmaTextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<PaliGemmaVisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("paligemma"));
    c.vocab_size = j.at("vocab_size").get<int>();
    c.ignore_index = j.value("ignore_index", -100);
    c.image_token_index = j.at("image_token_index").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.pad_token_id = j.value("pad_token_id", 0);
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

static mx::array gelu(const mx::array& x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// Gemma RMSNorm: rms_norm(x, 1.0 + weight, eps)
static mx::array gemma_rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    auto adjusted = mx::add(mx::array(1.0f), weight);
    return mx::fast::rms_norm(x, adjusted, eps);
}

// ── Vision Components (SigLip) ─────────────────────────────────────────

// -- Vision Embeddings --

PaliGemmaVisionEmbeddings::PaliGemmaVisionEmbeddings(const PaliGemmaVisionConfiguration& config)
    : patch_embedding_weight_(mx::zeros({config.hidden_size, config.patch_size,
                                          config.patch_size, config.num_channels})),
      position_embedding_weight_(mx::zeros({config.num_positions(), config.hidden_size})),
      patch_size_(config.patch_size),
      hidden_size_(config.hidden_size),
      num_positions_(config.num_positions())
{}

mx::array PaliGemmaVisionEmbeddings::operator()(const mx::array& x) {
    // x: [B, H, W, C] (already transposed from [B,C,H,W] by caller)
    int B = x.shape(0);
    int H = x.shape(1);
    int W = x.shape(2);

    // Apply Conv2d via reshaping into patches and matmul
    // Number of patches along each dimension
    int nH = H / patch_size_;
    int nW = W / patch_size_;
    int num_patches = nH * nW;

    // Reshape x into patches: [B, nH, patch_size, nW, patch_size, C]
    auto patches = mx::reshape(x, {B, nH, patch_size_, nW, patch_size_, -1});
    // Transpose to [B, nH, nW, patch_size, patch_size, C]
    patches = mx::transpose(patches, {0, 1, 3, 2, 4, 5});
    // Flatten patches: [B, num_patches, patch_size * patch_size * C]
    int kernel_elements = patch_size_ * patch_size_ * x.shape(3);
    patches = mx::reshape(patches, {B, num_patches, kernel_elements});

    // Flatten conv kernel: [out, kH*kW*in]
    auto flat_kernel = mx::reshape(patch_embedding_weight_, {hidden_size_, kernel_elements});
    // Matmul: [B, num_patches, kernel_elements] x [kernel_elements, out] -> [B, num_patches, out]
    auto patch_embeds = mx::matmul(patches, mx::transpose(flat_kernel));

    // Add positional embeddings
    return mx::add(patch_embeds, position_embedding_weight_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionEmbeddings::weight_map() {
    return {
        {"patch_embedding.weight", &patch_embedding_weight_},
        {"position_embedding.weight", &position_embedding_weight_},
    };
}

// -- Vision Attention --

PaliGemmaVisionAttention::PaliGemmaVisionAttention(int dims, int num_heads)
    : num_heads_(num_heads),
      head_dim_(dims / num_heads),
      scale_(std::pow(static_cast<float>(dims / num_heads), -0.5f)),
      wq_weight_(mx::zeros({dims, dims})),
      wq_bias_(mx::zeros({dims})),
      wk_weight_(mx::zeros({dims, dims})),
      wk_bias_(mx::zeros({dims})),
      wv_weight_(mx::zeros({dims, dims})),
      wv_bias_(mx::zeros({dims})),
      wo_weight_(mx::zeros({dims, dims})),
      wo_bias_(mx::zeros({dims}))
{}

mx::array PaliGemmaVisionAttention::operator()(const mx::array& x) {
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys    = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values  = linear_fwd(x, wv_weight_, &wv_bias_);

    // Reshape to [B, num_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});

    // No mask for vision self-attention (attend to all positions)
    auto output = mx::fast::scaled_dot_product_attention(queries, keys, values, scale_);

    // Reshape back to [B, L, dims]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"out_proj.weight", &wo_weight_}, {"out_proj.bias", &wo_bias_},
    };
}

// -- Vision MLP --

PaliGemmaVisionMLP::PaliGemmaVisionMLP(const PaliGemmaVisionConfiguration& config)
    : fc1_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      fc1_bias_(mx::zeros({config.intermediate_size})),
      fc2_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      fc2_bias_(mx::zeros({config.hidden_size}))
{}

mx::array PaliGemmaVisionMLP::operator()(const mx::array& x) {
    // fc1 -> GELU -> fc2
    return linear_fwd(gelu(linear_fwd(x, fc1_weight_, &fc1_bias_)), fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// -- Vision Encoder Layer --

PaliGemmaVisionEncoderLayer::PaliGemmaVisionEncoderLayer(const PaliGemmaVisionConfiguration& config)
    : attention_(config.hidden_size, config.num_attention_heads),
      mlp_(config),
      layer_norm1_weight_(mx::ones({config.hidden_size})),
      layer_norm1_bias_(mx::zeros({config.hidden_size})),
      layer_norm2_weight_(mx::ones({config.hidden_size})),
      layer_norm2_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array PaliGemmaVisionEncoderLayer::operator()(const mx::array& x) {
    // Pre-norm: h = x + attn(layernorm1(x))
    auto h = mx::add(x,
        attention_(mx::fast::layer_norm(x, layer_norm1_weight_, layer_norm1_bias_, eps_)));
    // Pre-norm: h = h + mlp(layernorm2(h))
    h = mx::add(h,
        mlp_(mx::fast::layer_norm(h, layer_norm2_weight_, layer_norm2_bias_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionEncoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["layer_norm1.weight"] = &layer_norm1_weight_;
    map["layer_norm1.bias"] = &layer_norm1_bias_;
    map["layer_norm2.weight"] = &layer_norm2_weight_;
    map["layer_norm2.bias"] = &layer_norm2_bias_;
    return map;
}

// -- Vision Encoder --

PaliGemmaVisionEncoder::PaliGemmaVisionEncoder(const PaliGemmaVisionConfiguration& config) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array PaliGemmaVisionEncoder::operator()(const mx::array& x) {
    auto h = x;
    for (auto& layer : layers_) {
        h = layer(h);
    }
    return h;
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- SigLip Vision Model --

PaliGemmaSigLipVisionModel::PaliGemmaSigLipVisionModel(const PaliGemmaVisionConfiguration& config)
    : embeddings_(config),
      encoder_(config),
      post_layernorm_weight_(mx::ones({config.hidden_size})),
      post_layernorm_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array PaliGemmaSigLipVisionModel::operator()(const mx::array& x) {
    auto h = embeddings_(x);
    h = encoder_(h);
    return mx::fast::layer_norm(h, post_layernorm_weight_, post_layernorm_bias_, eps_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaSigLipVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embeddings_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    map["post_layernorm.weight"] = &post_layernorm_weight_;
    map["post_layernorm.bias"] = &post_layernorm_bias_;
    return map;
}

// -- Vision Model Wrapper --

PaliGemmaVisionModel::PaliGemmaVisionModel(const PaliGemmaVisionConfiguration& config)
    : vision_model_(config),
      num_channels_(config.num_channels)
{}

mx::array PaliGemmaVisionModel::operator()(const mx::array& x) {
    // Input x: [B, C, H, W] from processor
    // Transpose to [B, H, W, C] for MLX conv/patch operations
    auto input = mx::transpose(x, {0, 2, 3, 1});
    return vision_model_(input);
}

std::unordered_map<std::string, mx::array> PaliGemmaVisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("patch_embedding.weight") != std::string::npos) {
            // PyTorch conv2d weight: [out, in, kH, kW]
            // MLX format: [out, kH, kW, in]
            if (v.ndim() == 4 && v.shape(1) == num_channels_) {
                sanitized.insert_or_assign(k, mx::transpose(v, {0, 2, 3, 1}));
            } else {
                sanitized.insert_or_assign(k, v);
            }
        } else {
            sanitized.insert_or_assign(k, v);
        }
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> PaliGemmaVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_model_.weight_map()) map["vision_model." + k] = v;
    return map;
}

// ── Language Components (Gemma-based) ──────────────────────────────────

// -- Language Attention --

PaliGemmaLanguageAttention::PaliGemmaLanguageAttention(const PaliGemmaTextConfiguration& args)
    : heads_(args.num_attention_heads),
      kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim()),
      scale_(std::pow(static_cast<float>(args.head_dim()), -0.5f)),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim()}))
{}

mx::array PaliGemmaLanguageAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    // Reshape to [B, num_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
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

std::unordered_map<std::string, mx::array*> PaliGemmaLanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

PaliGemmaLanguageMLP::PaliGemmaLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array PaliGemmaLanguageMLP::operator()(const mx::array& x) {
    // down(gelu(gate(x)) * up(x))
    return linear_fwd(mx::multiply(gelu(linear_fwd(x, gate_weight_)),
                                    linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Transformer Block --

PaliGemmaTransformerBlock::PaliGemmaTransformerBlock(const PaliGemmaTextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array PaliGemmaTransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Gemma RMSNorm: rms_norm(x, 1.0 + weight, eps)
    auto r = attention_(gemma_rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(gemma_rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> PaliGemmaTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

PaliGemmaLanguageModelInner::PaliGemmaLanguageModelInner(const PaliGemmaTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      hidden_scale_(std::sqrt(static_cast<float>(args.hidden_size)))
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array PaliGemmaLanguageModelInner::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    mx::array h = [&]() -> mx::array {
        if (input_embedding.has_value()) {
            return input_embedding.value();
        } else if (inputs.has_value()) {
            auto emb = mx::take(embed_tokens_weight_, inputs.value(), 0);
            // Gemma scales embeddings by sqrt(hidden_size)
            return mx::multiply(emb, mx::array(hidden_scale_));
        } else {
            throw std::runtime_error("Either inputs or input_embedding must be provided");
        }
    }();

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    // Gemma RMSNorm for final norm
    return gemma_rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array PaliGemmaLanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array PaliGemmaLanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> PaliGemmaLanguageModelInner::weight_map() {
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

PaliGemmaLanguageModel::PaliGemmaLanguageModel(const PaliGemmaTextConfiguration& args)
    : model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

LMOutput PaliGemmaLanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);
    // Gemma always uses tied embeddings
    out = model_.embed_as_linear(out);
    return LMOutput(out);
}

std::unordered_map<std::string, mx::array*> PaliGemmaLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

// ── Multimodal Projector ───────────────────────────────────────────────

PaliGemmaMultiModalProjector::PaliGemmaMultiModalProjector(const PaliGemmaVisionConfiguration& config)
    : weight_(mx::zeros({config.projection_dim, config.hidden_size})),
      bias_(mx::zeros({config.projection_dim}))
{}

mx::array PaliGemmaMultiModalProjector::operator()(const mx::array& x) {
    return linear_fwd(x, weight_, &bias_);
}

std::unordered_map<std::string, mx::array*> PaliGemmaMultiModalProjector::weight_map() {
    return {
        {"linear.weight", &weight_},
        {"linear.bias", &bias_},
    };
}

// ── Top-Level PaliGemma Model ──────────────────────────────────────────

PaliGemmaModel::PaliGemmaModel(const PaliGemmaConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config),
      multi_modal_projector_(config.vision_config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

PrepareResult PaliGemmaModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    if (!input.image.has_value()) {
        // Text-only: run through language model directly
        std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
        auto result = language_model_(input_ids, cache_ptr);
        return PrepareResult::logits(std::move(result));
    }

    // Get image pixels and run through vision tower
    auto pixel_values = input.image->pixels;
    auto vision_outputs = vision_tower_(pixel_values);

    // Project vision features to language model dimension
    auto image_features = multi_modal_projector_(vision_outputs);

    // Scale projected features by 1/sqrt(hidden_size)
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(config_.hidden_size));
    image_features = mx::multiply(image_features, mx::array(inv_scale));

    // Get text embeddings and scale by sqrt(hidden_size) (Gemma embedding scaling)
    auto input_embeds = language_model_.inner().embed_tokens(input_ids);
    float embed_scale = std::sqrt(static_cast<float>(config_.text_config.hidden_size));
    input_embeds = mx::multiply(input_embeds, mx::array(embed_scale));

    // Ensure image_features has batch dimension: [B, num_image_tokens, hidden_size]
    if (image_features.ndim() == 2) {
        image_features = mx::expand_dims(image_features, 0);
    }

    // Merge image tokens into text embeddings using masking
    // Create a mask where image_token_index positions are True
    auto image_mask = mx::equal(input_ids, mx::array(config_.image_token_index));
    // Expand mask for broadcasting: [B, L, 1]
    auto mask_expanded = mx::expand_dims(image_mask, -1);
    // Convert to float for where operation
    auto mask_float = mx::astype(mask_expanded, input_embeds.dtype());

    // Flatten image features to match the number of image token positions
    // image_features: [B, num_image_tokens, D]
    // We need to scatter them into the positions marked by image_token_index
    // Use a simple approach: for each batch, replace image_token positions with image features

    // Count image token positions per batch for reshaping
    // Construct the final embeddings using where: mask selects image vs text
    // Build a flat image feature tensor that aligns with all image_token positions
    int B = input_ids.shape(0);
    int L = input_ids.shape(1);
    int D = input_embeds.shape(-1);
    int num_image_tokens = image_features.shape(1);

    // Create an image embedding tensor of same shape as input_embeds
    // by scattering image_features into positions where image_token_index appears
    // Approach: reshape image_features to [B, num_image_tokens, D], pad to [B, L, D]
    // then use where to select

    // Build a sequential image tensor aligned with text positions
    // For each position marked as image_token, sequentially fill from image_features
    // This works because PaliGemma places image tokens contiguously at the start

    // Create index array for scatter: cumulative sum of mask gives sequential indices
    auto cum_mask = mx::cumsum(mx::astype(image_mask, mx::int32), 1);
    // cum_mask[b, l] = number of image tokens at or before position l
    // We want: at each image_token position, index = cum_mask - 1 (0-based)
    auto gather_indices = mx::subtract(cum_mask, mx::array(1, mx::int32));
    // Clamp to valid range
    gather_indices = mx::clip(gather_indices, mx::array(0, mx::int32),
                               mx::array(num_image_tokens - 1, mx::int32));

    // Gather image features using these indices: [B, L, D]
    // For each batch b and position l: image_expanded[b,l,:] = image_features[b, gather_indices[b,l], :]
    // Use advanced indexing: expand gather_indices to [B, L, 1] and broadcast
    auto idx_expanded = mx::expand_dims(gather_indices, -1);
    // Tile index to match feature dimension
    idx_expanded = mx::broadcast_to(idx_expanded, {B, L, D});

    // Gather along dim 1 from image_features
    // mx::take_along_axis: image_features is [B, num_image_tokens, D]
    // idx_expanded is [B, L, D] with values in [0, num_image_tokens)
    auto image_expanded = mx::take_along_axis(image_features, idx_expanded, 1);

    // Use where to merge: at image_token positions use image_expanded, otherwise text
    auto final_embeds = mx::where(mask_expanded, image_expanded, input_embeds);

    // Run through language model with pre-computed embeddings
    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, final_embeds);

    return PrepareResult::logits(std::move(result));
}

LMOutput PaliGemmaModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array PaliGemmaModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
PaliGemmaModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Remove rotary embedding inverse frequency keys (not needed, computed dynamically)
    std::unordered_map<std::string, mx::array> filtered;
    for (auto& [k, v] : weights) {
        if (k.find("rotary_emb.inv_freq") != std::string::npos) {
            continue;
        }
        filtered.insert_or_assign(k, v);
    }

    // Sanitize vision conv weights (PyTorch [out,in,kH,kW] -> MLX [out,kH,kW,in])
    return vision_tower_.sanitize(std::move(filtered));
}

void PaliGemmaModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> PaliGemmaModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    // Vision tower: prefix "vision_tower."
    for (auto& [k, v] : vision_tower_.weight_map())
        map["vision_tower." + k] = v;
    // Language model: prefix "language_model."
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;
    // Multi-modal projector: prefix "multi_modal_projector."
    for (auto& [k, v] : multi_modal_projector_.weight_map())
        map["multi_modal_projector." + k] = v;
    return map;
}

} // namespace mlx_lm

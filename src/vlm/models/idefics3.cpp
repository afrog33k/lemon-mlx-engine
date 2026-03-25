// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Idefics3.swift — Idefics3 VLM (SigLip vision + Llama language + connector)

#include <mlx-lm/vlm/models/idefics3.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Idefics3TextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("llama"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.value("num_hidden_layers", 32);
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
}

void from_json(const nlohmann::json& j, Idefics3VisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("siglip_vision_model"));
    c.num_hidden_layers = j.value("num_hidden_layers", 12);
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.value("intermediate_size", 3072);
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.image_size = j.at("image_size").get<int>();
    c.num_channels = j.value("num_channels", 3);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
}

void from_json(const nlohmann::json& j, Idefics3Configuration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<Idefics3TextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Idefics3VisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("idefics3"));
    c.ignore_index = j.value("ignore_index", -100);
    c.vocab_size = j.value("vocab_size", 128259);
    c.scale_factor = j.value("scale_factor", 2);
    c.image_token_id = j.value("image_token_id", 49153);
    // image_token_index defaults to image_token_id if not present
    c.image_token_index = j.value("image_token_index", c.image_token_id);
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

// GELU precise: x * 0.5 * (1 + erf(x / sqrt(2)))
static mx::array gelu(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// Check if a 4D array is already in MLX conv format [O,H,W,C] with O>=H, O>=W, H==W
static bool check_array_shape(const mx::array& arr) {
    if (arr.ndim() != 4) return false;
    int o = arr.shape(0), h = arr.shape(1), w = arr.shape(2);
    return (o >= h && o >= w && h == w);
}

// ── Vision Components (SigLip) ─────────────────────────────────────────

// -- Vision Embeddings --

Idefics3VisionEmbeddings::Idefics3VisionEmbeddings(const Idefics3VisionConfiguration& config)
    : patch_embedding_weight_(mx::zeros({config.hidden_size, config.patch_size,
                                          config.patch_size, config.num_channels})),
      position_embedding_weight_(mx::zeros({config.num_positions(), config.hidden_size})),
      patch_size_(config.patch_size),
      hidden_size_(config.hidden_size),
      num_positions_(config.num_positions())
{}

mx::array Idefics3VisionEmbeddings::operator()(const mx::array& x) {
    // x: [B, H, W, C] (already in BHWC format)
    int B = x.shape(0);
    int H = x.shape(1);
    int W = x.shape(2);

    // Apply Conv2d via reshaping into patches and matmul
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

std::unordered_map<std::string, mx::array*> Idefics3VisionEmbeddings::weight_map() {
    return {
        {"patch_embedding.weight", &patch_embedding_weight_},
        {"position_embedding.weight", &position_embedding_weight_},
    };
}

// -- Vision Attention --

Idefics3VisionAttention::Idefics3VisionAttention(int dims, int num_heads)
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

mx::array Idefics3VisionAttention::operator()(const mx::array& x) {
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

std::unordered_map<std::string, mx::array*> Idefics3VisionAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"out_proj.weight", &wo_weight_}, {"out_proj.bias", &wo_bias_},
    };
}

// -- Vision MLP --

Idefics3VisionMLP::Idefics3VisionMLP(const Idefics3VisionConfiguration& config)
    : fc1_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      fc1_bias_(mx::zeros({config.intermediate_size})),
      fc2_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      fc2_bias_(mx::zeros({config.hidden_size}))
{}

mx::array Idefics3VisionMLP::operator()(const mx::array& x) {
    // fc1 -> GELU(precise) -> fc2
    return linear_fwd(gelu(linear_fwd(x, fc1_weight_, &fc1_bias_)), fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> Idefics3VisionMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// -- Vision Encoder Layer --

Idefics3VisionEncoderLayer::Idefics3VisionEncoderLayer(const Idefics3VisionConfiguration& config)
    : attention_(config.hidden_size, config.num_attention_heads),
      mlp_(config),
      layer_norm1_weight_(mx::ones({config.hidden_size})),
      layer_norm1_bias_(mx::zeros({config.hidden_size})),
      layer_norm2_weight_(mx::ones({config.hidden_size})),
      layer_norm2_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array Idefics3VisionEncoderLayer::operator()(const mx::array& x) {
    // Pre-norm: h = x + attn(layernorm1(x))
    auto h = mx::add(x,
        attention_(mx::fast::layer_norm(x, layer_norm1_weight_, layer_norm1_bias_, eps_)));
    // Pre-norm: h = h + mlp(layernorm2(h))
    h = mx::add(h,
        mlp_(mx::fast::layer_norm(h, layer_norm2_weight_, layer_norm2_bias_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> Idefics3VisionEncoderLayer::weight_map() {
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

Idefics3VisionEncoder::Idefics3VisionEncoder(const Idefics3VisionConfiguration& config) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array Idefics3VisionEncoder::operator()(const mx::array& x) {
    auto h = x;
    for (auto& layer : layers_) {
        h = layer(h);
    }
    return h;
}

std::unordered_map<std::string, mx::array*> Idefics3VisionEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Vision Inner Model (SigLip) --

Idefics3VisionInnerModel::Idefics3VisionInnerModel(const Idefics3VisionConfiguration& config)
    : embeddings_(config),
      encoder_(config),
      post_layernorm_weight_(mx::ones({config.hidden_size})),
      post_layernorm_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array Idefics3VisionInnerModel::operator()(const mx::array& x) {
    auto h = embeddings_(x);
    h = encoder_(h);
    return mx::fast::layer_norm(h, post_layernorm_weight_, post_layernorm_bias_, eps_);
}

std::unordered_map<std::string, mx::array*> Idefics3VisionInnerModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embeddings_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    map["post_layernorm.weight"] = &post_layernorm_weight_;
    map["post_layernorm.bias"] = &post_layernorm_bias_;
    return map;
}

// -- Vision Model Wrapper --

Idefics3VisionModel::Idefics3VisionModel(const Idefics3VisionConfiguration& config)
    : vision_model_(config),
      num_channels_(config.num_channels)
{}

mx::array Idefics3VisionModel::operator()(const mx::array& x) {
    // Input x: [B, H, W, C] (already in BHWC format from processor)
    return vision_model_(x);
}

std::unordered_map<std::string, mx::array> Idefics3VisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("position_ids") != std::string::npos) {
            continue; // Remove position_ids
        } else if (k.find("patch_embedding.weight") != std::string::npos) {
            // If already in MLX format [O,H,W,C], keep as is;
            // otherwise transpose from [O,C,H,W] -> [O,H,W,C]
            if (check_array_shape(v)) {
                sanitized.insert_or_assign(k, v);
            } else {
                sanitized.insert_or_assign(k, mx::transpose(v, {0, 2, 3, 1}));
            }
        } else {
            sanitized.insert_or_assign(k, v);
        }
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> Idefics3VisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_model_.weight_map()) map["vision_model." + k] = v;
    return map;
}

// ── Connector (Pixel Shuffle + MLP) ────────────────────────────────────

// -- Connector MLP --

Idefics3ConnectorMLP::Idefics3ConnectorMLP(const Idefics3Configuration& config)
    : proj_weight_(mx::zeros({config.text_config.hidden_size,
                               config.vision_config.hidden_size * config.scale_factor * config.scale_factor}))
{}

mx::array Idefics3ConnectorMLP::operator()(const mx::array& x) {
    // Linear projection (no bias)
    return linear_fwd(x, proj_weight_);
}

std::unordered_map<std::string, mx::array*> Idefics3ConnectorMLP::weight_map() {
    return {
        {"proj.weight", &proj_weight_},
    };
}

// -- Connector --

Idefics3Connector::Idefics3Connector(const Idefics3Configuration& config)
    : modality_projection_(config),
      scale_factor_(config.scale_factor)
{}

mx::array Idefics3Connector::pixel_shuffle(const mx::array& x, int scale_factor) {
    int B = x.shape(0);
    int seq = x.shape(1);
    int embed_dim = x.shape(2);
    int side = static_cast<int>(std::sqrt(static_cast<float>(seq)));

    // x: [B, seq, embed_dim]
    auto reshaped = mx::reshape(x, {B, side, side, embed_dim});
    // [B, side, side/sf, embed_dim*sf]
    reshaped = mx::reshape(reshaped, {B, side, side / scale_factor, embed_dim * scale_factor});
    // Transpose: [B, side/sf, side, embed_dim*sf]
    reshaped = mx::transpose(reshaped, {0, 2, 1, 3});
    // [B, side/sf, side/sf, embed_dim*sf*sf]
    reshaped = mx::reshape(reshaped, {B, side / scale_factor, side / scale_factor,
                                       embed_dim * scale_factor * scale_factor});
    // Transpose: [B, side/sf, side/sf, embed_dim*sf*sf]
    reshaped = mx::transpose(reshaped, {0, 2, 1, 3});
    // [B, seq/(sf*sf), embed_dim*sf*sf]
    reshaped = mx::reshape(reshaped, {B, seq / (scale_factor * scale_factor),
                                       embed_dim * scale_factor * scale_factor});
    return reshaped;
}

mx::array Idefics3Connector::operator()(const mx::array& image_hidden_states) {
    auto shuffled = pixel_shuffle(image_hidden_states, scale_factor_);
    return modality_projection_(shuffled);
}

std::unordered_map<std::string, mx::array*> Idefics3Connector::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : modality_projection_.weight_map())
        map["modality_projection." + k] = v;
    return map;
}

// ── Language Components (Llama-style) ──────────────────────────────────

// -- Language Attention --

Idefics3LanguageAttention::Idefics3LanguageAttention(const Idefics3TextConfiguration& args)
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

mx::array Idefics3LanguageAttention::operator()(
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

std::unordered_map<std::string, mx::array*> Idefics3LanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

Idefics3LanguageMLP::Idefics3LanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Idefics3LanguageMLP::operator()(const mx::array& x) {
    // down(swiglu(gate(x), up(x)))
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Idefics3LanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Transformer Block --

Idefics3TransformerBlock::Idefics3TransformerBlock(const Idefics3TextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Idefics3TransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Standard Llama RMSNorm (no 1+weight trick like Gemma)
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Idefics3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

Idefics3LanguageModelInner::Idefics3LanguageModelInner(const Idefics3TextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Idefics3LanguageModelInner::operator()(
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

    // Standard Llama RMSNorm for final norm
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Idefics3LanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array Idefics3LanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Idefics3LanguageModelInner::weight_map() {
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

Idefics3LanguageModel::Idefics3LanguageModel(const Idefics3TextConfiguration& args)
    : model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    // Only create lm_head when tie_word_embeddings is false
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

LMOutput Idefics3LanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);
    if (lm_head_weight_.has_value()) {
        out = mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    } else {
        out = model_.embed_as_linear(out);
    }
    return LMOutput(out);
}

std::unordered_map<std::string, mx::array*> Idefics3LanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Top-Level Idefics3 Model ───────────────────────────────────────────

Idefics3Model::Idefics3Model(const Idefics3Configuration& config)
    : config_(config),
      vision_model_(config.vision_config),
      language_model_(config.text_config),
      connector_(config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

PrepareResult Idefics3Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    // Get text embeddings
    auto inputs_embeds = language_model_.inner().embed_tokens(input_ids);

    if (input.image.has_value()) {
        // Get image pixels and run through vision tower
        auto pixel_values = input.image->pixels;
        auto vision_outputs = vision_model_(pixel_values);

        // Run through connector (pixel shuffle + projection)
        // Match dtype with input embeddings
        auto image_features = connector_(mx::astype(vision_outputs, inputs_embeds.dtype()));

        // Merge image features into text embeddings at image_token positions
        // image_features: [num_images, num_image_tokens_per_image, D]
        // inputs_embeds: [B, seq_len, D]

        int B = input_ids.shape(0);
        int L = input_ids.shape(1);
        int D = inputs_embeds.shape(-1);

        // Find image token positions and replace them with vision features
        // image_features has shape [num_images, chunk_size, D]
        // We need to scatter each chunk into the correct positions

        // Create mask for image token positions
        auto image_mask = mx::equal(input_ids, mx::array(config_.image_token_index));

        // Use cumulative sum approach to map positions to image feature indices
        auto cum_mask = mx::cumsum(mx::astype(image_mask, mx::int32), 1);
        auto gather_indices = mx::subtract(cum_mask, mx::array(1, mx::int32));

        int num_image_tokens = image_features.shape(0) * image_features.shape(1);
        // Flatten image_features from [num_images, tokens_per_image, D] -> [total_tokens, D]
        auto flat_image_features = mx::reshape(image_features, {num_image_tokens, D});
        // Add batch dim: [1, total_tokens, D]
        flat_image_features = mx::expand_dims(flat_image_features, 0);

        // Clamp gather indices to valid range
        gather_indices = mx::clip(gather_indices, mx::array(0, mx::int32),
                                   mx::array(std::max(num_image_tokens - 1, 0), mx::int32));

        // Expand gather_indices for broadcasting: [B, L, 1] -> [B, L, D]
        auto idx_expanded = mx::expand_dims(gather_indices, -1);
        idx_expanded = mx::broadcast_to(idx_expanded, {B, L, D});

        // Gather image features aligned with text positions
        auto image_expanded = mx::take_along_axis(flat_image_features, idx_expanded, 1);

        // Use where to merge: at image_token positions use image features, otherwise text
        auto mask_expanded = mx::expand_dims(image_mask, -1);
        inputs_embeds = mx::where(mask_expanded, image_expanded, inputs_embeds);
    }

    // Run through language model with pre-computed embeddings
    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, inputs_embeds);

    return PrepareResult::logits(std::move(result));
}

LMOutput Idefics3Model::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Idefics3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
Idefics3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Step 1: Rename keys to match internal structure
    // Strip "model." prefix, or prepend "language_model." to "lm_head." keys
    std::unordered_map<std::string, mx::array> renamed;
    for (auto& [k, v] : weights) {
        std::string new_key = k;
        if (new_key.substr(0, 6) == "model.") {
            new_key = new_key.substr(6); // strip "model." prefix
        } else if (new_key.substr(0, 8) == "lm_head.") {
            new_key = "language_model." + new_key;
        }
        renamed.insert_or_assign(new_key, v);
    }

    // Step 2: Rename "text_model." to "language_model."
    std::unordered_map<std::string, mx::array> final_weights;
    for (auto& [k, v] : renamed) {
        if (k.substr(0, 11) == "text_model.") {
            std::string suffix = k.substr(11);
            final_weights.insert_or_assign("language_model." + suffix, v);
        } else {
            final_weights.insert_or_assign(k, v);
        }
    }

    // Step 3: Remove rotary_emb.inv_freq keys
    std::unordered_map<std::string, mx::array> filtered;
    for (auto& [k, v] : final_weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            continue;
        }
        filtered.insert_or_assign(k, v);
    }

    // Step 4: Sanitize vision conv weights (PyTorch [O,C,H,W] -> MLX [O,H,W,C])
    return vision_model_.sanitize(std::move(filtered));
}

void Idefics3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Idefics3Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    // Vision model: prefix "vision_model."
    for (auto& [k, v] : vision_model_.weight_map())
        map["vision_model." + k] = v;
    // Language model: prefix "language_model."
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;
    // Connector: prefix "connector."
    for (auto& [k, v] : connector_.weight_map())
        map["connector." + k] = v;
    return map;
}

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/vlm/models/qwen2_vl.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Qwen2VLTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("qwen2_vl"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }
}

void from_json(const nlohmann::json& j, Qwen2VLVisionConfiguration& c) {
    c.depth = j.at("depth").get<int>();
    c.embed_dim = j.at("embed_dim").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_heads = j.at("num_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.mlp_ratio = j.value("mlp_ratio", 4.0f);
    c.in_channels = j.value("in_channels", 3);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
    c.spatial_patch_size = j.at("spatial_patch_size").get<int>();
    c.spatial_merge_size = j.value("spatial_merge_size", 2);
    c.temporal_patch_size = j.value("temporal_patch_size", 2);
}

void from_json(const nlohmann::json& j, Qwen2VLBaseConfiguration& c) {
    c.model_type = j.at("model_type").get<std::string>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.image_token_id = j.at("image_token_id").get<int>();
    c.video_token_id = j.at("video_token_id").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
}

void from_json(const nlohmann::json& j, Qwen2VLConfiguration& c) {
    // Vision config is a sub-dictionary
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Qwen2VLVisionConfiguration>();
    }
    // Text and base configs are overlaid in the top level
    c.text_config = j.get<Qwen2VLTextConfiguration>();
    c.base_config = j.get<Qwen2VLBaseConfiguration>();
}

void from_json(const nlohmann::json& j, Qwen2VLProcessorConfiguration& c) {
    if (j.contains("image_mean")) c.image_mean = j["image_mean"].get<std::vector<float>>();
    if (j.contains("image_std")) c.image_std = j["image_std"].get<std::vector<float>>();
    c.merge_size = j.value("merge_size", 2);
    c.patch_size = j.value("patch_size", 14);
    c.temporal_patch_size = j.value("temporal_patch_size", 2);

    // Handle both flat and nested size config
    if (j.contains("min_pixels")) c.min_pixels = j["min_pixels"].get<int>();
    if (j.contains("max_pixels")) c.max_pixels = j["max_pixels"].get<int>();
    if (j.contains("size")) {
        auto& sz = j["size"];
        if (sz.contains("min_pixels")) c.min_pixels = sz["min_pixels"].get<int>();
        if (sz.contains("max_pixels")) c.max_pixels = sz["max_pixels"].get<int>();
    }
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

// Apply multimodal rotary position embedding for vision
static mx::array apply_vision_rotary_emb(const mx::array& tensor, const mx::array& freqs) {
    auto cos_f = mx::cos(freqs);
    auto sin_f = mx::sin(freqs);

    // Expand dims and tile for the full head dimension
    cos_f = mx::expand_dims(cos_f, 1);  // [S, 1, D/2]
    cos_f = mx::concatenate({cos_f, cos_f}, -1); // [S, 1, D]
    cos_f = mx::expand_dims(cos_f, 0);  // [1, S, 1, D]

    sin_f = mx::expand_dims(sin_f, 1);
    sin_f = mx::concatenate({sin_f, sin_f}, -1);
    sin_f = mx::expand_dims(sin_f, 0);

    return mx::add(mx::multiply(tensor, cos_f),
                   mx::multiply(qwen_vl::rotate_half(tensor), sin_f));
}

// ── Vision Components ──────────────────────────────────────────────────

// -- PatchMerger --

Qwen2VLPatchMerger::Qwen2VLPatchMerger(int dimensions, int context_dimensions, int spatial_merge_size)
    : hidden_size_(context_dimensions * spatial_merge_size * spatial_merge_size),
      ln_q_weight_(mx::ones({context_dimensions})),
      ln_q_bias_(mx::zeros({context_dimensions})),
      mlp_0_weight_(mx::zeros({hidden_size_, hidden_size_})),
      mlp_0_bias_(mx::zeros({hidden_size_})),
      mlp_2_weight_(mx::zeros({dimensions, hidden_size_})),
      mlp_2_bias_(mx::zeros({dimensions})),
      eps_(1e-6f)
{}

mx::array Qwen2VLPatchMerger::operator()(const mx::array& x) {
    // LayerNorm then reshape to merge spatial patches
    auto normed = mx::fast::layer_norm(x, ln_q_weight_, ln_q_bias_, eps_);
    auto merged = mx::reshape(normed, {-1, hidden_size_});
    // MLP: Linear -> GELU -> Linear
    merged = linear_fwd(merged, mlp_0_weight_, &mlp_0_bias_);
    merged = gelu(merged);
    merged = linear_fwd(merged, mlp_2_weight_, &mlp_2_bias_);
    return merged;
}

std::unordered_map<std::string, mx::array*> Qwen2VLPatchMerger::weight_map() {
    return {
        {"ln_q.weight", &ln_q_weight_}, {"ln_q.bias", &ln_q_bias_},
        {"mlp.0.weight", &mlp_0_weight_}, {"mlp.0.bias", &mlp_0_bias_},
        {"mlp.2.weight", &mlp_2_weight_}, {"mlp.2.bias", &mlp_2_bias_},
    };
}

// -- Vision Attention --

Qwen2VLVisionAttention::Qwen2VLVisionAttention(int dims, int num_heads)
    : num_heads_(num_heads),
      scale_(std::pow(static_cast<float>(dims / num_heads), -0.5f)),
      qkv_weight_(mx::zeros({3 * dims, dims})),
      qkv_bias_(mx::zeros({3 * dims})),
      proj_weight_(mx::zeros({dims, dims})),
      proj_bias_(mx::zeros({dims}))
{}

mx::array Qwen2VLVisionAttention::operator()(
    const mx::array& x,
    const std::vector<THW>& frames,
    const mx::array& rotary_pos_emb)
{
    int seq_len = x.shape(0);
    int B = frames[0].t;
    int L = seq_len / B;

    auto qkv_out = linear_fwd(x, qkv_weight_, &qkv_bias_);
    int dim = qkv_out.shape(-1) / 3;

    // Split into Q, K, V
    auto q = mx::slice(qkv_out, {0, 0}, {seq_len, dim});
    auto k = mx::slice(qkv_out, {0, dim}, {seq_len, 2 * dim});
    auto v = mx::slice(qkv_out, {0, 2 * dim}, {seq_len, 3 * dim});

    int head_dim = dim / num_heads_;
    q = mx::reshape(q, {seq_len, num_heads_, head_dim});
    k = mx::reshape(k, {seq_len, num_heads_, head_dim});
    v = mx::reshape(v, {seq_len, num_heads_, head_dim});

    // Apply vision rotary embedding
    q = apply_vision_rotary_emb(q, rotary_pos_emb);
    k = apply_vision_rotary_emb(k, rotary_pos_emb);

    // Reshape for attention: [B, num_heads, L, head_dim]
    q = mx::transpose(mx::reshape(q, {B, L, num_heads_, head_dim}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, num_heads_, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, num_heads_, head_dim}), {0, 2, 1, 3});

    auto output = mx::fast::scaled_dot_product_attention(q, k, v, scale_);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {seq_len, -1});

    return linear_fwd(output, proj_weight_, &proj_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen2VLVisionAttention::weight_map() {
    return {
        {"qkv.weight", &qkv_weight_}, {"qkv.bias", &qkv_bias_},
        {"proj.weight", &proj_weight_}, {"proj.bias", &proj_bias_},
    };
}

// -- Vision MLP --

Qwen2VLVisionMLP::Qwen2VLVisionMLP(int dimensions, int hidden_dimensions)
    : fc1_weight_(mx::zeros({hidden_dimensions, dimensions})),
      fc1_bias_(mx::zeros({hidden_dimensions})),
      fc2_weight_(mx::zeros({dimensions, hidden_dimensions})),
      fc2_bias_(mx::zeros({dimensions}))
{}

mx::array Qwen2VLVisionMLP::operator()(const mx::array& x) {
    return linear_fwd(gelu(linear_fwd(x, fc1_weight_, &fc1_bias_)), fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen2VLVisionMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// -- Vision Block --

Qwen2VLVisionBlock::Qwen2VLVisionBlock(const Qwen2VLVisionConfiguration& config)
    : attention_(config.embed_dim, config.num_heads),
      mlp_(config.embed_dim, static_cast<int>(config.embed_dim * config.mlp_ratio)),
      norm1_weight_(mx::ones({config.embed_dim})),
      norm1_bias_(mx::zeros({config.embed_dim})),
      norm2_weight_(mx::ones({config.embed_dim})),
      norm2_bias_(mx::zeros({config.embed_dim})),
      eps_(config.layer_norm_eps)
{}

mx::array Qwen2VLVisionBlock::operator()(
    const mx::array& hidden_states,
    const std::vector<THW>& frames,
    const mx::array& rotary_pos_emb)
{
    auto h = mx::add(hidden_states,
        attention_(mx::fast::layer_norm(hidden_states, norm1_weight_, norm1_bias_, eps_),
                   frames, rotary_pos_emb));
    h = mx::add(h, mlp_(mx::fast::layer_norm(h, norm2_weight_, norm2_bias_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> Qwen2VLVisionBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["norm1.weight"] = &norm1_weight_;
    map["norm1.bias"] = &norm1_bias_;
    map["norm2.weight"] = &norm2_weight_;
    map["norm2.bias"] = &norm2_bias_;
    return map;
}

// -- Vision Model --

Qwen2VLVisionModel::Qwen2VLVisionModel(const Qwen2VLVisionConfiguration& config)
    : patch_embed_proj_weight_(mx::zeros({config.embed_dim, config.temporal_patch_size,
                                           config.patch_size, config.patch_size, config.in_channels})),
      patch_size_(config.patch_size),
      temporal_patch_size_(config.temporal_patch_size),
      in_channels_(config.in_channels),
      embed_dim_(config.embed_dim),
      rotary_pos_emb_(config.embed_dim / config.num_heads / 2, 10000.0f),
      merger_(config.hidden_size, config.embed_dim, config.spatial_merge_size),
      spatial_merge_size_(config.spatial_merge_size)
{
    blocks_.reserve(config.depth);
    for (int i = 0; i < config.depth; ++i)
        blocks_.emplace_back(config);
}

mx::array Qwen2VLVisionModel::patch_embed(const mx::array& hidden_states) {
    // Reshape input: [N, C, T, H, W] -> [N, T, H, W, C] for Conv3d
    auto x = mx::reshape(hidden_states,
        {-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    // Move channels to last: [N, T, H, W, C]
    x = mx::transpose(x, {0, 2, 3, 4, 1});
    // Apply Conv3d as matmul with reshaped kernel
    // Conv3d kernel: [out, T, H, W, C] -> flatten spatial dims
    int kernel_elements = temporal_patch_size_ * patch_size_ * patch_size_ * in_channels_;
    auto flat_x = mx::reshape(x, {-1, kernel_elements});
    auto flat_w = mx::reshape(patch_embed_proj_weight_, {embed_dim_, kernel_elements});
    auto out = mx::matmul(flat_x, mx::transpose(flat_w));
    return out; // [N, embed_dim]
}

mx::array Qwen2VLVisionModel::compute_rotary_pos_emb(const std::vector<THW>& frames) {
    std::vector<mx::array> position_ids;

    for (const auto& frame : frames) {
        int t = frame.t, h = frame.h, w = frame.w;

        // Height position IDs
        auto h_ids = mx::expand_dims(mx::arange(0, h, mx::int32), 1); // [h, 1]
        h_ids = mx::repeat(h_ids, w, 1); // [h, w]
        h_ids = mx::reshape(h_ids, {h / spatial_merge_size_, spatial_merge_size_,
                                     w / spatial_merge_size_, spatial_merge_size_});
        h_ids = mx::transpose(h_ids, {0, 2, 1, 3});
        h_ids = mx::flatten(h_ids);

        // Width position IDs
        auto w_ids = mx::expand_dims(mx::arange(0, w, mx::int32), 0); // [1, w]
        w_ids = mx::repeat(w_ids, h, 0); // [h, w]
        w_ids = mx::reshape(w_ids, {h / spatial_merge_size_, spatial_merge_size_,
                                     w / spatial_merge_size_, spatial_merge_size_});
        w_ids = mx::transpose(w_ids, {0, 2, 1, 3});
        w_ids = mx::flatten(w_ids);

        // Stack [h_ids, w_ids] -> [h*w, 2] and tile for temporal dimension
        auto stacked = mx::stack({h_ids, w_ids}, -1); // [h*w, 2]
        // Tile for temporal: repeat t times
        std::vector<mx::array> tiled_frames;
        for (int ti = 0; ti < t; ++ti) tiled_frames.push_back(stacked);
        position_ids.push_back(mx::concatenate(tiled_frames, 0));
    }

    auto indices = mx::concatenate(position_ids, 0); // [total, 2]

    // Get max frame size for computing full frequency table
    int max_size = 0;
    for (const auto& f : frames) max_size = std::max(max_size, std::max(f.h, f.w));

    auto full_freqs = rotary_pos_emb_(max_size); // [max_size, D/2]

    // Gather: full_freqs[indices] -> [total, 2, D/2] -> reshape to [total, D]
    auto result = mx::take(full_freqs, indices, 0);
    return mx::reshape(result, {indices.shape(0), -1});
}

mx::array Qwen2VLVisionModel::operator()(const mx::array& hidden_states, const std::vector<THW>& frames) {
    auto h = patch_embed(hidden_states);
    auto rope = compute_rotary_pos_emb(frames);

    for (auto& block : blocks_) {
        h = block(h, frames, rope);
    }

    return merger_(h);
}

std::unordered_map<std::string, mx::array> Qwen2VLVisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("position_id") != std::string::npos) {
            continue; // Remove unused position_ids
        }
        if (k.find("patch_embed.proj.weight") != std::string::npos) {
            // PyTorch conv weight: [out, in, T, H, W]
            // MLX conv weight: [out, T, H, W, in]
            if (v.ndim() == 5 && v.shape(-1) != in_channels_) {
                sanitized.insert_or_assign(k, mx::transpose(v, {0, 2, 3, 4, 1}));
            } else {
                sanitized.insert_or_assign(k, v);
            }
        } else {
            sanitized.insert_or_assign(k, v);
        }
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> Qwen2VLVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["patch_embed.proj.weight"] = &patch_embed_proj_weight_;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto prefix = "blocks." + std::to_string(i) + ".";
        for (auto& [k, v] : blocks_[i].weight_map()) map[prefix + k] = v;
    }
    for (auto& [k, v] : merger_.weight_map()) map["merger." + k] = v;
    return map;
}

// ── Language Components ────────────────────────────────────────────────

// -- Language Attention --

Qwen2VLLanguageAttention::Qwen2VLLanguageAttention(const Qwen2VLTextConfiguration& args)
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

mx::array Qwen2VLLanguageAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values = linear_fwd(x, wv_weight_, &wv_bias_);

    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen2VLLanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

Qwen2VLLanguageMLP::Qwen2VLLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen2VLLanguageMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen2VLLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Decoder Layer --

Qwen2VLDecoderLayer::Qwen2VLDecoderLayer(const Qwen2VLTextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Qwen2VLDecoderLayer::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen2VLDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

Qwen2VLLanguageModelInner::Qwen2VLLanguageModelInner(const Qwen2VLTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Qwen2VLLanguageModelInner::operator()(
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

mx::array Qwen2VLLanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array Qwen2VLLanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen2VLLanguageModelInner::weight_map() {
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

Qwen2VLLanguageModel::Qwen2VLLanguageModel(const Qwen2VLTextConfiguration& args)
    : model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

LMOutput Qwen2VLLanguageModel::operator()(
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

std::unordered_map<std::string, mx::array*> Qwen2VLLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Top-Level Model ────────────────────────────────────────────────────

Qwen2VLModel::Qwen2VLModel(const Qwen2VLConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array Qwen2VLModel::input_embeddings(
    const mx::array& input_ids,
    const mx::array* pixel_values,
    const std::vector<THW>* frames)
{
    if (!pixel_values || !frames) {
        // Text-only: expand dims for batch
        auto ids = input_ids;
        if (ids.ndim() == 1) ids = mx::expand_dims(ids, 0);
        return language_model_.inner().embed_tokens(ids);
    }

    // Get text embeddings
    auto input_embeds = language_model_.inner().embed_tokens(input_ids);

    // Get vision features from vision tower
    auto hidden_states = vision_tower_(*pixel_values, *frames);

    if (hidden_states.ndim() == 2) {
        hidden_states = mx::expand_dims(hidden_states, 0);
    }

    // Merge vision features into text embeddings at special token positions
    return qwen_vl::merge_input_ids_with_image_features(
        input_ids, input_embeds, hidden_states,
        config_.base_config.image_token_id,
        config_.base_config.video_token_id);
}

PrepareResult Qwen2VLModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    // Collect image and video pixels/frames
    const mx::array* all_pixels = nullptr;
    std::vector<THW> all_frames;
    mx::array combined_pixels = mx::array(0.0f);

    if (input.image.has_value()) {
        combined_pixels = input.image->pixels;
        if (input.image->frames.has_value()) {
            all_frames.insert(all_frames.end(),
                input.image->frames->begin(), input.image->frames->end());
        }
    }

    if (input.video.has_value()) {
        if (all_frames.empty()) {
            combined_pixels = input.video->pixels;
        } else {
            combined_pixels = mx::concatenate({combined_pixels, input.video->pixels}, 0);
        }
        if (input.video->frames.has_value()) {
            all_frames.insert(all_frames.end(),
                input.video->frames->begin(), input.video->frames->end());
        }
    }

    if (!all_frames.empty()) {
        all_pixels = &combined_pixels;
    }

    auto input_embeds = input_embeddings(
        input.text.tokens, all_pixels,
        all_frames.empty() ? nullptr : &all_frames);

    // Run language model with pre-computed embeddings
    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, input_embeds);

    return PrepareResult::logits(std::move(result));
}

LMOutput Qwen2VLModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Qwen2VLModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array> Qwen2VLModel::sanitize_impl(
    std::unordered_map<std::string, mx::array> weights)
{
    // Rename keys: visual -> vision_tower, model -> language_model.model
    std::unordered_map<std::string, mx::array> renamed;
    for (auto& [k, v] : weights) {
        std::string key = k;
        if (key.find("vision_tower") == std::string::npos) {
            // Replace "visual" with "vision_tower"
            size_t pos = key.find("visual");
            if (pos != std::string::npos) {
                key.replace(pos, 6, "vision_tower");
            }
        }
        if (key.find("language_model") == std::string::npos) {
            size_t pos = key.find("lm_head");
            if (pos != std::string::npos) {
                key = "language_model.lm_head" + key.substr(pos + 7);
            } else {
                size_t mpos = key.find("model.");
                if (mpos != std::string::npos && mpos == 0) {
                    key = "language_model." + key;
                }
            }
        }
        renamed.insert_or_assign(key, v);
    }

    // Sanitize vision weights (conv format conversion)
    return vision_tower_.sanitize(std::move(renamed));
}

void Qwen2VLModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen2VLModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_tower_.weight_map()) map["vision_tower." + k] = v;
    for (auto& [k, v] : language_model_.weight_map()) map["language_model." + k] = v;
    return map;
}

} // namespace mlx_lm

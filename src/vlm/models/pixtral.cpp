// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/pixtral

#include <mlx-lm/vlm/models/pixtral.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, PixtralVisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("pixtral"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.image_size = j.at("image_size").get<int>();
    c.num_channels = j.value("num_channels", 3);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
    c.head_dim = j.value("head_dim", 0);
    c.rope_theta = j.value("rope_theta", 10000.0f);
}

void from_json(const nlohmann::json& j, PixtralTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("pixtral"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.head_dim = j.value("head_dim", 0);
    c.max_position_embeddings = j.value("max_position_embeddings", 0);
    c.num_key_value_heads = j.value("num_key_value_heads", 0);
    c.rope_theta = j.value("rope_theta", 1000000000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.use_qk_norm = j.value("use_qk_norm", false);

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

void from_json(const nlohmann::json& j, PixtralConfiguration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<PixtralTextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<PixtralVisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("pixtral"));
    c.ignore_index = j.value("ignore_index", -100);

    // Support both image_token_index and image_token_id
    if (j.contains("image_token_index")) {
        c.image_token_index = j["image_token_index"].get<int>();
    } else if (j.contains("image_token_id")) {
        c.image_token_index = j["image_token_id"].get<int>();
    } else {
        c.image_token_index = 10;
    }

    c.vision_feature_select_strategy = j.value("vision_feature_select_strategy", std::string("full"));
    c.vision_feature_layer = j.value("vision_feature_layer", -1);
    c.vocab_size = j.value("vocab_size", 32000);
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

static mx::array gelu(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// rotate_half: split last dim in half, negate second half, swap
static mx::array rotate_half(const mx::array& x) {
    int half_dim = x.shape(-1) / 2;
    // x1 = x[..., :half_dim], x2 = x[..., half_dim:]
    mx::Shape start_s(x.ndim(), 0);
    mx::Shape stop_s(x.shape());

    auto stop1 = stop_s;
    stop1.back() = half_dim;
    auto x1 = mx::slice(x, start_s, stop1);

    auto start2 = start_s;
    start2.back() = half_dim;
    auto x2 = mx::slice(x, start2, stop_s);

    return mx::concatenate({mx::negative(x2), x1}, -1);
}

// Check if conv weight is in OIHW format (needs transpose to OHWI)
static bool check_array_shape_oihw(const mx::array& arr) {
    if (arr.ndim() != 4) return false;
    int o = arr.shape(0), h = arr.shape(1), w = arr.shape(2);
    return (o >= h && o >= w && h == w);
}

// Generate position IDs in meshgrid pattern: for each (h, w) -> h * max_width + w
static mx::array position_ids_in_meshgrid(int patch_height, int patch_width, int max_width) {
    std::vector<int32_t> positions;
    positions.reserve(patch_height * patch_width);
    for (int h = 0; h < patch_height; ++h) {
        for (int w = 0; w < patch_width; ++w) {
            positions.push_back(static_cast<int32_t>(h * max_width + w));
        }
    }
    return mx::array(positions.data(), {static_cast<int>(positions.size())}, mx::int32);
}

// Generate block-diagonal attention mask for separate images
static mx::array generate_block_attention_mask(
    const std::vector<int>& patch_counts, int batch_size, mx::Dtype dtype)
{
    int seq_len = 0;
    for (auto c : patch_counts) seq_len += c;

    std::vector<float> mask_data(seq_len * seq_len, -1e9f);

    int start = 0;
    for (auto count : patch_counts) {
        int end = start + count;
        for (int row = start; row < end; ++row) {
            for (int col = start; col < end; ++col) {
                mask_data[row * seq_len + col] = 0.0f;
            }
        }
        start = end;
    }

    auto mask_arr = mx::array(mask_data.data(), {seq_len, seq_len}, mx::float32);
    // Expand to [1, 1, seq_len, seq_len] then broadcast to [batch_size, 1, seq_len, seq_len]
    mask_arr = mx::expand_dims(mx::expand_dims(mask_arr, 0), 0);
    mask_arr = mx::broadcast_to(mask_arr, {batch_size, 1, seq_len, seq_len});
    return mx::astype(mask_arr, dtype);
}

// ── Vision Components ──────────────────────────────────────────────────

// -- Rotary Embedding --

PixtralVisionRotaryEmbedding::PixtralVisionRotaryEmbedding(const PixtralVisionConfiguration& config)
    : dim_(config.effective_head_dim()),
      inv_freq_(mx::array(0.0f))
{
    float base = config.rope_theta;
    int max_patches_per_side = config.image_size / config.patch_size;
    int half_dim = dim_ / 2;

    // Create base frequencies: 1 / (base ^ (arange(0, dim, 2) / dim))
    std::vector<float> freq_vals(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freq_vals[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(dim_));
    }
    auto freqs = mx::array(freq_vals.data(), {half_dim}, mx::float32);

    // Split freqs by alternating indices for h and w
    // Even indices go to h, odd indices go to w
    int quarter = half_dim / 2;
    std::vector<int> even_indices, odd_indices;
    for (int i = 0; i < half_dim; ++i) {
        if (i % 2 == 0) even_indices.push_back(i);
        else odd_indices.push_back(i);
    }

    auto even_idx = mx::array(even_indices.data(), {static_cast<int>(even_indices.size())}, mx::int32);
    auto odd_idx = mx::array(odd_indices.data(), {static_cast<int>(odd_indices.size())}, mx::int32);

    auto freqs_h_sel = mx::take(freqs, even_idx, 0);  // (quarter,)
    auto freqs_w_sel = mx::take(freqs, odd_idx, 0);    // (quarter,)

    // Create position ranges
    auto h_pos = mx::astype(mx::arange(0, max_patches_per_side, 1), mx::float32);
    auto w_pos = mx::astype(mx::arange(0, max_patches_per_side, 1), mx::float32);

    // Outer products: freqs_H[h, :] = h_pos[h] * freqs_h_sel[:]
    auto freqs_H = mx::matmul(
        mx::reshape(h_pos, {max_patches_per_side, 1}),
        mx::reshape(freqs_h_sel, {1, quarter})
    );  // (max_patches, quarter)

    auto freqs_W = mx::matmul(
        mx::reshape(w_pos, {max_patches_per_side, 1}),
        mx::reshape(freqs_w_sel, {1, quarter})
    );  // (max_patches, quarter)

    // Tile: freqs_H[h, :] repeated across all widths -> (max_patches, max_patches, quarter)
    auto tiled_H = mx::broadcast_to(
        mx::expand_dims(freqs_H, 1),
        {max_patches_per_side, max_patches_per_side, quarter}
    );

    auto tiled_W = mx::broadcast_to(
        mx::expand_dims(freqs_W, 0),
        {max_patches_per_side, max_patches_per_side, quarter}
    );

    // Concatenate along last dim -> (max_patches, max_patches, half_dim)
    auto inv_freq_2d = mx::concatenate({tiled_H, tiled_W}, -1);

    // Reshape to (max_patches^2, half_dim)
    inv_freq_2d = mx::reshape(inv_freq_2d, {max_patches_per_side * max_patches_per_side, half_dim});

    // Duplicate for full dim: (max_patches^2, dim)
    inv_freq_ = mx::concatenate({inv_freq_2d, inv_freq_2d}, -1);
    mx::eval(inv_freq_);
}

std::pair<mx::array, mx::array>
PixtralVisionRotaryEmbedding::operator()(const mx::array& x, const mx::array& position_ids) {
    auto gathered = mx::take(inv_freq_, position_ids, 0);  // (num_positions, dim)
    auto cos_f = mx::astype(mx::cos(gathered), x.dtype());
    auto sin_f = mx::astype(mx::sin(gathered), x.dtype());
    return {cos_f, sin_f};
}

// -- Vision Attention --

PixtralVisionAttention::PixtralVisionAttention(const PixtralVisionConfiguration& config)
    : num_heads_(config.num_attention_heads),
      head_dim_(config.effective_head_dim()),
      scale_(std::pow(static_cast<float>(config.effective_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.hidden_size, config.hidden_size})),
      wk_weight_(mx::zeros({config.hidden_size, config.hidden_size})),
      wv_weight_(mx::zeros({config.hidden_size, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.hidden_size}))
{}

mx::array PixtralVisionAttention::operator()(
    const mx::array& x,
    const std::pair<mx::array, mx::array>& position_embeddings,
    const AttentionMask& mask)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    // Reshape to [B, num_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply 2D rotary position embeddings
    // cos/sin shape: (L, head_dim), expand to (1, 1, L, head_dim) for broadcasting
    auto cos_f = mx::expand_dims(mx::expand_dims(position_embeddings.first, 0), 0);   // (1, 1, L, dim)
    auto sin_f = mx::expand_dims(mx::expand_dims(position_embeddings.second, 0), 0);

    queries = mx::add(mx::multiply(queries, cos_f), mx::multiply(rotate_half(queries), sin_f));
    keys    = mx::add(mx::multiply(keys, cos_f), mx::multiply(rotate_half(keys), sin_f));

    // Manual attention with mask support
    auto attn_weights = mx::multiply(
        mx::matmul(queries, mx::transpose(keys, {0, 1, 3, 2})),
        mx::array(scale_)
    );

    if (mask.has_array()) {
        attn_weights = mx::add(attn_weights, mask.as_array());
    }

    attn_weights = mx::softmax(attn_weights, -1);
    auto output = mx::matmul(attn_weights, values);

    // Reshape back to [B, L, hidden_size]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> PixtralVisionAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Vision MLP --

PixtralVisionMLP::PixtralVisionMLP(const PixtralVisionConfiguration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size}))
{}

mx::array PixtralVisionMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> PixtralVisionMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Vision Encoder Layer --

PixtralVisionEncoderLayer::PixtralVisionEncoderLayer(const PixtralVisionConfiguration& config)
    : attention_(config),
      feed_forward_(config),
      attention_norm_weight_(mx::ones({config.hidden_size})),
      ffn_norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array PixtralVisionEncoderLayer::operator()(
    const mx::array& x,
    const std::pair<mx::array, mx::array>& position_embeddings,
    const AttentionMask& mask)
{
    auto y = mx::fast::rms_norm(x, attention_norm_weight_, rms_norm_eps_);
    auto attn_out = attention_(y, position_embeddings, mask);
    auto h = mx::add(x, attn_out);
    auto ffn_out = feed_forward_(mx::fast::rms_norm(h, ffn_norm_weight_, rms_norm_eps_));
    return mx::add(h, ffn_out);
}

std::unordered_map<std::string, mx::array*> PixtralVisionEncoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attention." + k] = v;
    for (auto& [k, v] : feed_forward_.weight_map()) map["feed_forward." + k] = v;
    map["attention_norm.weight"] = &attention_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;
    return map;
}

// -- Vision Encoder --

PixtralVisionEncoder::PixtralVisionEncoder(const PixtralVisionConfiguration& config) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

std::pair<mx::array, std::vector<mx::array>>
PixtralVisionEncoder::operator()(
    const mx::array& x,
    const std::pair<mx::array, mx::array>& position_embeddings,
    const AttentionMask& mask,
    bool output_hidden_states)
{
    std::vector<mx::array> hidden_states;
    if (output_hidden_states) {
        hidden_states.push_back(x);
    }

    auto h = x;
    for (auto& layer : layers_) {
        h = layer(h, position_embeddings, mask);
        if (output_hidden_states) {
            hidden_states.push_back(h);
        }
    }

    return {h, hidden_states};
}

std::unordered_map<std::string, mx::array*> PixtralVisionEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Pixtral Vision Model Inner --

PixtralVisionModelInner::PixtralVisionModelInner(const PixtralVisionConfiguration& config)
    : patch_conv_weight_(mx::zeros({config.hidden_size, config.patch_size,
                                     config.patch_size, config.num_channels})),
      ln_pre_weight_(mx::ones({config.hidden_size})),
      transformer_(config),
      rope_(config),
      config_(config)
{}

std::pair<mx::array, std::vector<mx::array>>
PixtralVisionModelInner::operator()(const mx::array& x, bool output_hidden_states) {
    // x is NHWC format: (batch, height, width, channels)
    auto input = mx::astype(x, patch_conv_weight_.dtype());

    int B = input.shape(0);
    int H = input.shape(1);
    int W = input.shape(2);
    int C = input.shape(3);
    int ps = config_.patch_size;

    // Apply Conv2d via reshape + matmul (stride = patch_size)
    int nH = H / ps;
    int nW = W / ps;
    int num_patches = nH * nW;

    // Reshape into patches: [B, nH, ps, nW, ps, C] -> [B, nH, nW, ps, ps, C]
    auto patches = mx::reshape(input, {B, nH, ps, nW, ps, C});
    patches = mx::transpose(patches, {0, 1, 3, 2, 4, 5});
    int kernel_elements = ps * ps * C;
    patches = mx::reshape(patches, {B, num_patches, kernel_elements});

    // Flatten conv kernel and apply as matmul
    auto flat_kernel = mx::reshape(patch_conv_weight_, {config_.hidden_size, kernel_elements});
    auto patch_embeds = mx::matmul(patches, mx::transpose(flat_kernel));

    // Apply ln_pre (RMSNorm)
    patch_embeds = mx::fast::rms_norm(patch_embeds, ln_pre_weight_, config_.rms_norm_eps);

    // Compute position IDs in meshgrid pattern
    int max_width = config_.image_size / config_.patch_size;
    auto position_ids = position_ids_in_meshgrid(nH, nW, max_width);
    auto position_embedding = rope_(patch_embeds, position_ids);

    // Generate block attention mask
    int patches_per_image = nH * nW;
    std::vector<int> patch_counts(B, patches_per_image);
    auto mask = generate_block_attention_mask(patch_counts, B, patch_embeds.dtype());

    // Run through transformer
    auto [encoded, hidden_states] = transformer_(
        patch_embeds, position_embedding, AttentionMask::from_array(mask), output_hidden_states);

    return {encoded, hidden_states};
}

std::unordered_map<std::string, mx::array*> PixtralVisionModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["patch_conv.weight"] = &patch_conv_weight_;
    map["ln_pre.weight"] = &ln_pre_weight_;
    for (auto& [k, v] : transformer_.weight_map()) map["transformer." + k] = v;
    return map;
}

// -- Vision Model Wrapper --

PixtralVisionModel::PixtralVisionModel(const PixtralVisionConfiguration& config)
    : vision_model_(config),
      num_channels_(config.num_channels),
      num_encoder_layers_(config.num_hidden_layers)
{}

PixtralVisionModel::VisionOutput
PixtralVisionModel::operator()(const mx::array& x, bool output_hidden_states) {
    // Input x: [B, C, H, W] from processor -> transpose to [B, H, W, C]
    auto input = mx::transpose(x, {0, 2, 3, 1});

    auto [encoded, hidden_states] = vision_model_(input, output_hidden_states);
    auto embeddings = hidden_states.empty() ? encoded : hidden_states.front();

    return {encoded, embeddings, hidden_states};
}

std::unordered_map<std::string, mx::array> PixtralVisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("position_ids") != std::string::npos) {
            continue;
        }
        if (k.find("patch_conv.weight") != std::string::npos ||
            k.find("patch_embedding.weight") != std::string::npos) {
            if (check_array_shape_oihw(v)) {
                sanitized.insert_or_assign(k, v);
            } else {
                // Transpose from OIHW to OHWI
                sanitized.insert_or_assign(k, mx::transpose(v, {0, 2, 3, 1}));
            }
        } else {
            sanitized.insert_or_assign(k, v);
        }
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> PixtralVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_model_.weight_map()) map["vision_model." + k] = v;
    return map;
}

size_t PixtralVisionModel::num_encoder_layers() const {
    return num_encoder_layers_;
}

// ── Language Components ────────────────────────────────────────────────

// -- Language Attention --

PixtralLanguageAttention::PixtralLanguageAttention(const PixtralTextConfiguration& args)
    : heads_(args.num_attention_heads),
      kv_heads_(args.effective_num_kv_heads()),
      head_dim_(args.effective_head_dim()),
      scale_(std::pow(static_cast<float>(args.effective_head_dim()), -0.5f)),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional),
      rope_scale_(1.0f),
      use_qk_norm_(args.use_qk_norm),
      rms_norm_eps_(args.rms_norm_eps),
      wq_weight_(mx::zeros({args.num_attention_heads * args.effective_head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.effective_num_kv_heads() * args.effective_head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.effective_num_kv_heads() * args.effective_head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.effective_head_dim()})),
      q_norm_weight_(args.use_qk_norm ? mx::ones({args.effective_head_dim()}) : mx::array(0.0f)),
      k_norm_weight_(args.use_qk_norm ? mx::ones({args.effective_head_dim()}) : mx::array(0.0f))
{
    // Handle rope scaling
    if (args.rope_scaling.has_value()) {
        auto& scaling = args.rope_scaling.value();
        auto type_it = scaling.find("type");
        auto factor_it = scaling.find("factor");
        if (type_it != scaling.end() && type_it->second.is_string() &&
            type_it->second.as_string() == "linear" &&
            factor_it != scaling.end() && factor_it->second.is_float()) {
            rope_scale_ = 1.0f / factor_it->second.as_float();
        }
    }
}

mx::array PixtralLanguageAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Optional QK norms
    if (use_qk_norm_) {
        queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
        keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    }

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);
    keys    = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> PixtralLanguageAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (use_qk_norm_) {
        map["q_norm.weight"] = &q_norm_weight_;
        map["k_norm.weight"] = &k_norm_weight_;
    }
    return map;
}

// -- Language MLP --

PixtralLanguageMLP::PixtralLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array PixtralLanguageMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> PixtralLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Transformer Block --

PixtralTransformerBlock::PixtralTransformerBlock(const PixtralTextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array PixtralTransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> PixtralTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

PixtralLanguageModelInner::PixtralLanguageModelInner(const PixtralTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array PixtralLanguageModelInner::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    mx::array h = mx::array(0.0f);
    if (input_embedding.has_value()) {
        h = input_embedding.value();
    } else if (inputs.has_value()) {
        h = mx::take(embed_tokens_weight_, inputs.value(), 0);
    } else {
        throw std::runtime_error("Either inputs or input_embedding must be provided");
    }

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array PixtralLanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array PixtralLanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> PixtralLanguageModelInner::weight_map() {
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

PixtralLanguageModel::PixtralLanguageModel(const PixtralTextConfiguration& args)
    : model_(args),
      tie_word_embeddings_(args.tie_word_embeddings)
{
    kv_heads_.resize(args.num_hidden_layers, args.effective_num_kv_heads());
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

LMOutput PixtralLanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);
    if (tie_word_embeddings_) {
        out = model_.embed_as_linear(out);
    } else if (lm_head_weight_.has_value()) {
        out = mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    }
    return LMOutput(out);
}

std::unordered_map<std::string, mx::array*> PixtralLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Multimodal Projector ───────────────────────────────────────────────

PixtralMultiModalProjector::PixtralMultiModalProjector(int vision_hidden_size, int text_hidden_size)
    : linear_1_weight_(mx::zeros({text_hidden_size, vision_hidden_size})),
      linear_1_bias_(mx::zeros({text_hidden_size})),
      linear_2_weight_(mx::zeros({text_hidden_size, text_hidden_size})),
      linear_2_bias_(mx::zeros({text_hidden_size}))
{}

mx::array PixtralMultiModalProjector::operator()(const mx::array& x) {
    auto result = linear_fwd(x, linear_1_weight_, &linear_1_bias_);
    result = gelu(result);
    result = linear_fwd(result, linear_2_weight_, &linear_2_bias_);
    return result;
}

std::unordered_map<std::string, mx::array*> PixtralMultiModalProjector::weight_map() {
    return {
        {"linear_1.weight", &linear_1_weight_},
        {"linear_1.bias", &linear_1_bias_},
        {"linear_2.weight", &linear_2_weight_},
        {"linear_2.bias", &linear_2_bias_},
    };
}

// ── Top-Level Pixtral Model ───────────────────────────────────────────

PixtralModel::PixtralModel(const PixtralConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config),
      multi_modal_projector_(config.vision_config.hidden_size, config.text_config.hidden_size),
      vision_feature_layer_(config.vision_feature_layer)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array PixtralModel::get_input_embeddings(
    const mx::array& input_ids,
    const mx::array* pixel_values)
{
    if (!pixel_values) {
        return language_model_.inner().embed_tokens(input_ids);
    }

    auto inputs_embeds = language_model_.inner().embed_tokens(input_ids);

    // Process through vision tower with hidden states
    auto vision_out = vision_tower_(*pixel_values, /*output_hidden_states=*/true);

    // Select features from specified layer
    auto& hidden_states = vision_out.hidden_states;
    int layer_index = vision_feature_layer_ < 0
        ? static_cast<int>(hidden_states.size()) + vision_feature_layer_
        : vision_feature_layer_;

    auto selected_features = hidden_states[layer_index];

    // Project to text space
    auto image_features = multi_modal_projector_(selected_features);

    // Merge embeddings: find image token positions and interleave
    int num_image_patches = image_features.shape(1);

    // Assuming batch size 1
    // Find positions of image_token_index in input_ids
    auto input_ids_flat = input_ids.ndim() > 1
        ? mx::reshape(input_ids, {-1})
        : input_ids;

    // Build merged embeddings by finding image token positions
    // We iterate through token IDs, collecting text segments and image patches
    int seq_len = input_ids_flat.shape(0);
    mx::eval(input_ids_flat);
    const int32_t* ids_data = input_ids_flat.data<int32_t>();

    std::vector<int> image_positions;
    for (int i = 0; i < seq_len; ++i) {
        if (ids_data[i] == static_cast<int32_t>(config_.image_token_index)) {
            image_positions.push_back(i);
        }
    }

    // inputs_embeds shape: [1, seq_len, hidden] or [seq_len, hidden]
    auto embeds = inputs_embeds;
    if (embeds.ndim() == 2) {
        embeds = mx::expand_dims(embeds, 0);
    }

    std::vector<mx::array> final_embeddings;
    int start_idx = 0;

    for (size_t i = 0; i < image_positions.size(); ++i) {
        int pos = image_positions[i];
        // Text segment before this image token
        if (pos > start_idx) {
            final_embeddings.push_back(
                mx::slice(embeds, {0, start_idx, 0}, {embeds.shape(0), pos, embeds.shape(2)}));
        }
        // Image patch embedding
        if (static_cast<int>(i) < num_image_patches) {
            final_embeddings.push_back(
                mx::slice(image_features, {0, static_cast<int>(i), 0},
                          {image_features.shape(0), static_cast<int>(i) + 1, image_features.shape(2)}));
        }
        start_idx = pos + 1;
    }

    // Remaining text after last image token
    if (start_idx < seq_len) {
        final_embeddings.push_back(
            mx::slice(embeds, {0, start_idx, 0}, {embeds.shape(0), seq_len, embeds.shape(2)}));
    }

    if (final_embeddings.empty()) {
        return embeds;
    }

    return mx::concatenate(final_embeddings, 1);
}

PrepareResult PixtralModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    const mx::array* pixel_values = nullptr;
    mx::array pixels = mx::array(0.0f);
    if (input.image.has_value()) {
        pixels = input.image->pixels;
        pixel_values = &pixels;
    }

    auto embeddings = get_input_embeddings(input_ids, pixel_values);

    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, embeddings);

    return PrepareResult::logits(std::move(result));
}

LMOutput PixtralModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array PixtralModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
PixtralModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    std::unordered_map<std::string, mx::array> new_weights;

    for (auto& [key, value] : weights) {
        std::string new_key = key;

        // Transform keys to match model structure
        if (key.find("vision_tower") != std::string::npos &&
            key.find("vision_model") == std::string::npos) {
            if (key.find("transformer") != std::string::npos ||
                key.find("patch_conv") != std::string::npos ||
                key.find("ln_pre") != std::string::npos) {
                auto pos = new_key.find("vision_tower");
                if (pos != std::string::npos) {
                    new_key.replace(pos, 12, "vision_tower.vision_model");
                }
            }
        } else if (key.find("vision_encoder") != std::string::npos &&
                   key.find("vision_tower") == std::string::npos) {
            if (key.find("transformer") != std::string::npos ||
                key.find("patch_conv") != std::string::npos ||
                key.find("ln_pre") != std::string::npos) {
                auto pos = new_key.find("model.vision_encoder");
                if (pos != std::string::npos) {
                    new_key.replace(pos, 20, "vision_tower.vision_model");
                }
            }
        } else if (key.find("model.language_model") != std::string::npos &&
                   key.find("language_model.model") == std::string::npos) {
            auto pos = new_key.find("model.language_model");
            if (pos != std::string::npos) {
                new_key.replace(pos, 20, "language_model.model");
            }
        } else if (key.find("lm_head") != std::string::npos &&
                   key.find("language_model") == std::string::npos) {
            auto pos = new_key.find("lm_head");
            if (pos != std::string::npos) {
                new_key.replace(pos, 7, "language_model.lm_head");
            }
        } else if (key.find("model.vision_projection") != std::string::npos) {
            auto pos = new_key.find("model.vision_projection");
            if (pos != std::string::npos) {
                new_key.replace(pos, 23, "multi_modal_projector");
            }
        }

        // Skip rotary embeddings
        if (new_key.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            continue;
        }

        if (new_weights.find(new_key) == new_weights.end()) {
            new_weights.insert_or_assign(new_key, value);
        }
    }

    // Sanitize vision conv weights
    return vision_tower_.sanitize(std::move(new_weights));
}

void PixtralModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> PixtralModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_tower_.weight_map())
        map["vision_tower." + k] = v;
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;
    for (auto& [k, v] : multi_modal_projector_.weight_map())
        map["multi_modal_projector." + k] = v;
    return map;
}

} // namespace mlx_lm

// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Qwen3VL.swift — Qwen3 Vision-Language Model

#include <mlx-lm/vlm/models/qwen3_vl.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Qwen3VLRoPEScaling& c) {
    c.type = j.value("type", std::string("default"));
    c.mrope_interleaved = j.value("mrope_interleaved", false);
    if (j.contains("mrope_section")) {
        c.mrope_section = j["mrope_section"].get<std::vector<int>>();
    } else {
        c.mrope_section = {24, 20, 20};
    }
}

void from_json(const nlohmann::json& j, Qwen3VLTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("qwen3_vl"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.head_dim = j.at("head_dim").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.max_position_embeddings = j.value("max_position_embeddings", 32768);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);
    c.attention_bias = j.value("attention_bias", false);
    c.hidden_act = j.value("hidden_act", std::string("silu"));

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        Qwen3VLRoPEScaling scaling;
        from_json(j["rope_scaling"], scaling);
        c.rope_scaling = scaling;
    }
}

void from_json(const nlohmann::json& j, Qwen3VLVisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("qwen3_vl"));
    c.depth = j.at("depth").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.out_hidden_size = j.at("out_hidden_size").get<int>();
    c.num_heads = j.at("num_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.spatial_merge_size = j.value("spatial_merge_size", 2);
    c.temporal_patch_size = j.value("temporal_patch_size", 2);
    c.num_position_embeddings = j.at("num_position_embeddings").get<int>();
    c.in_channels = j.value("in_channels", 3);
    c.hidden_act = j.value("hidden_act", std::string("gelu"));
    if (j.contains("deepstack_visual_indexes") && !j["deepstack_visual_indexes"].is_null()) {
        c.deepstack_visual_indexes = j["deepstack_visual_indexes"].get<std::vector<int>>();
    }
}

void from_json(const nlohmann::json& j, Qwen3VLBaseConfiguration& c) {
    c.model_type = j.at("model_type").get<std::string>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.image_token_id = j.value("image_token_id", 151655);
    c.video_token_id = j.value("video_token_id", 151656);
    c.vision_start_token_id = j.value("vision_start_token_id", 151652);
    c.vision_end_token_id = j.value("vision_end_token_id", 151653);
    c.vision_token_id = j.value("vision_token_id", 151654);
}

void from_json(const nlohmann::json& j, Qwen3VLConfiguration& c) {
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Qwen3VLVisionConfiguration>();
    }
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<Qwen3VLTextConfiguration>();
    } else {
        c.text_config = j.get<Qwen3VLTextConfiguration>();
    }
    c.base_config = j.get<Qwen3VLBaseConfiguration>();
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

// GELU with fast approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static mx::array gelu_fast(const mx::array& x) {
    auto x3 = mx::multiply(mx::multiply(x, x), x);
    auto inner = mx::multiply(
        mx::array(0.7978845608028654f),  // sqrt(2/pi)
        mx::add(x, mx::multiply(mx::array(0.044715f), x3)));
    return mx::multiply(
        mx::multiply(x, mx::array(0.5f)),
        mx::add(mx::array(1.0f), mx::tanh(inner)));
}

// GELU (precise): x * 0.5 * (1 + erf(x / sqrt(2)))
static mx::array gelu_precise(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// Apply rotary embedding to vision tensors (same pattern as Qwen2VL)
static mx::array apply_vision_rotary_emb(const mx::array& tensor, const mx::array& freqs) {
    auto cos_f = mx::cos(freqs);
    auto sin_f = mx::sin(freqs);

    // [S, D/2] -> [S, 1, D] -> [1, S, 1, D]
    cos_f = mx::expand_dims(cos_f, 1);
    cos_f = mx::concatenate({cos_f, cos_f}, -1);
    cos_f = mx::expand_dims(cos_f, 0);

    sin_f = mx::expand_dims(sin_f, 1);
    sin_f = mx::concatenate({sin_f, sin_f}, -1);
    sin_f = mx::expand_dims(sin_f, 0);

    return mx::add(mx::multiply(tensor, cos_f),
                   mx::multiply(qwen_vl::rotate_half(tensor), sin_f));
}

// Apply multimodal rotary embedding for language model
static std::pair<mx::array, mx::array> apply_multimodal_rotary(
    const mx::array& q, const mx::array& k,
    const mx::array& cos_vals, const mx::array& sin_vals)
{
    // cos/sin: [B, L, D] -> [B, 1, L, D] for broadcasting with [B, H, L, D]
    auto cos_expanded = mx::expand_dims(cos_vals, 1);
    auto sin_expanded = mx::expand_dims(sin_vals, 1);

    auto q_embed = mx::add(mx::multiply(q, cos_expanded),
                           mx::multiply(qwen_vl::rotate_half(q), sin_expanded));
    auto k_embed = mx::add(mx::multiply(k, cos_expanded),
                           mx::multiply(qwen_vl::rotate_half(k), sin_expanded));
    return {q_embed, k_embed};
}

// ── Vision Components ──────────────────────────────────────────────────

// -- PatchEmbed --

Qwen3VLPatchEmbed::Qwen3VLPatchEmbed(int patch_size, int temporal_patch_size,
                                       int in_channels, int hidden_size)
    : proj_weight_(mx::zeros({hidden_size, temporal_patch_size, patch_size, patch_size, in_channels})),
      proj_bias_(mx::zeros({hidden_size})),
      patch_size_(patch_size),
      temporal_patch_size_(temporal_patch_size),
      in_channels_(in_channels),
      hidden_size_(hidden_size)
{}

mx::array Qwen3VLPatchEmbed::operator()(const mx::array& x) {
    // Reshape input: flatten -> [N, C, T, H, W] -> [N, T, H, W, C]
    auto states = mx::reshape(x, {-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    states = mx::transpose(states, {0, 2, 3, 4, 1}); // Move channels to last

    // Conv3d as matmul: flatten spatial dims
    int kernel_elements = temporal_patch_size_ * patch_size_ * patch_size_ * in_channels_;
    auto flat_x = mx::reshape(states, {-1, kernel_elements});
    auto flat_w = mx::reshape(proj_weight_, {hidden_size_, kernel_elements});
    auto out = mx::add(mx::matmul(flat_x, mx::transpose(flat_w)), proj_bias_);
    return out;
}

std::unordered_map<std::string, mx::array*> Qwen3VLPatchEmbed::weight_map() {
    return {
        {"proj.weight", &proj_weight_},
        {"proj.bias", &proj_bias_},
    };
}

// -- PatchMerger --

static int qwen3vl_merger_hidden(const Qwen3VLVisionConfiguration& config) {
    return config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
}

Qwen3VLPatchMerger::Qwen3VLPatchMerger(const Qwen3VLVisionConfiguration& config,
                                         bool use_post_shuffle_norm)
    : hidden_size_(qwen3vl_merger_hidden(config)),
      use_post_shuffle_norm_(use_post_shuffle_norm),
      norm_weight_(mx::ones({use_post_shuffle_norm ? qwen3vl_merger_hidden(config) : config.hidden_size})),
      norm_bias_(mx::zeros({use_post_shuffle_norm ? qwen3vl_merger_hidden(config) : config.hidden_size})),
      linear_fc1_weight_(mx::zeros({qwen3vl_merger_hidden(config), qwen3vl_merger_hidden(config)})),
      linear_fc1_bias_(mx::zeros({qwen3vl_merger_hidden(config)})),
      linear_fc2_weight_(mx::zeros({config.out_hidden_size, qwen3vl_merger_hidden(config)})),
      linear_fc2_bias_(mx::zeros({config.out_hidden_size})),
      eps_(1e-6f)
{}

mx::array Qwen3VLPatchMerger::operator()(const mx::array& x) {
    auto states = x;
    if (use_post_shuffle_norm_) {
        // Deepstack: reshape first, then norm
        states = mx::reshape(states, {-1, hidden_size_});
    }
    states = mx::fast::layer_norm(states, norm_weight_, norm_bias_, eps_);
    states = mx::reshape(states, {-1, hidden_size_});
    states = linear_fwd(states, linear_fc1_weight_, &linear_fc1_bias_);
    states = gelu_precise(states); // PatchMerger uses standard GELU
    states = linear_fwd(states, linear_fc2_weight_, &linear_fc2_bias_);
    return states;
}

std::unordered_map<std::string, mx::array*> Qwen3VLPatchMerger::weight_map() {
    return {
        {"norm.weight", &norm_weight_}, {"norm.bias", &norm_bias_},
        {"linear_fc1.weight", &linear_fc1_weight_}, {"linear_fc1.bias", &linear_fc1_bias_},
        {"linear_fc2.weight", &linear_fc2_weight_}, {"linear_fc2.bias", &linear_fc2_bias_},
    };
}

// -- Vision Attention --

Qwen3VLVisionAttention::Qwen3VLVisionAttention(int dims, int num_heads)
    : num_heads_(num_heads),
      head_dim_(dims / num_heads),
      scale_(std::pow(static_cast<float>(dims / num_heads), -0.5f)),
      qkv_weight_(mx::zeros({3 * dims, dims})),
      qkv_bias_(mx::zeros({3 * dims})),
      proj_weight_(mx::zeros({dims, dims}))
{}

mx::array Qwen3VLVisionAttention::operator()(
    const mx::array& x,
    const mx::array& cu_seqlens,
    const mx::array& rotary_pos_emb)
{
    int seq_len = x.shape(0);

    // QKV projection
    auto qkv_out = linear_fwd(x, qkv_weight_, &qkv_bias_);

    // Reshape to [seq, 3, heads, head_dim] then transpose to [3, seq, heads, head_dim]
    qkv_out = mx::reshape(qkv_out, {seq_len, 3, num_heads_, head_dim_});
    qkv_out = mx::transpose(qkv_out, {1, 0, 2, 3});

    // Split into Q, K, V
    int dim = num_heads_ * head_dim_;
    auto q = mx::slice(qkv_out, {0, 0, 0, 0}, {1, seq_len, num_heads_, head_dim_});
    q = mx::squeeze(q, 0); // [seq, heads, head_dim]
    auto k = mx::slice(qkv_out, {1, 0, 0, 0}, {2, seq_len, num_heads_, head_dim_});
    k = mx::squeeze(k, 0);
    auto v = mx::slice(qkv_out, {2, 0, 0, 0}, {3, seq_len, num_heads_, head_dim_});
    v = mx::squeeze(v, 0);

    // Apply vision rotary embedding
    q = apply_vision_rotary_emb(q, rotary_pos_emb);
    k = apply_vision_rotary_emb(k, rotary_pos_emb);

    // Reshape for attention: [1, heads, seq, head_dim]
    q = mx::transpose(mx::reshape(q, {1, seq_len, num_heads_, head_dim_}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {1, seq_len, num_heads_, head_dim_}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {1, seq_len, num_heads_, head_dim_}), {0, 2, 1, 3});

    // Build block-diagonal attention mask from cumulative sequence lengths
    auto mask = mx::multiply(
        mx::ones({1, seq_len, seq_len}, q.dtype()),
        mx::array(-1e9f, q.dtype()));

    // Extract cu_seqlens values to build block-diagonal mask
    mx::eval(cu_seqlens);
    auto seqlens_data = cu_seqlens.data<int32_t>();
    int num_seqlens = cu_seqlens.shape(0);

    for (int idx = 1; idx < num_seqlens; ++idx) {
        int start = seqlens_data[idx - 1];
        int end = seqlens_data[idx];
        if (start < end) {
            auto zeros_block = mx::zeros({1, end - start, end - start}, q.dtype());
            mask = mx::slice_update(mask, zeros_block,
                                     mx::Shape{0, start, start},
                                     mx::Shape{1, end, end});
        }
    }

    auto output = sdpa(q, k, v, scale_, AttentionMask::from_array(mask));
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {seq_len, -1});

    return linear_fwd(output, proj_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3VLVisionAttention::weight_map() {
    return {
        {"qkv.weight", &qkv_weight_}, {"qkv.bias", &qkv_bias_},
        {"proj.weight", &proj_weight_},
    };
}

// -- Vision MLP --

Qwen3VLVisionMLP::Qwen3VLVisionMLP(int dimensions, int hidden_dimensions)
    : linear_fc1_weight_(mx::zeros({hidden_dimensions, dimensions})),
      linear_fc1_bias_(mx::zeros({hidden_dimensions})),
      linear_fc2_weight_(mx::zeros({dimensions, hidden_dimensions})),
      linear_fc2_bias_(mx::zeros({dimensions}))
{}

mx::array Qwen3VLVisionMLP::operator()(const mx::array& x) {
    // GELU(fast) for vision MLP
    return linear_fwd(gelu_fast(linear_fwd(x, linear_fc1_weight_, &linear_fc1_bias_)),
                      linear_fc2_weight_, &linear_fc2_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen3VLVisionMLP::weight_map() {
    return {
        {"linear_fc1.weight", &linear_fc1_weight_}, {"linear_fc1.bias", &linear_fc1_bias_},
        {"linear_fc2.weight", &linear_fc2_weight_}, {"linear_fc2.bias", &linear_fc2_bias_},
    };
}

// -- Vision Block --

Qwen3VLVisionBlock::Qwen3VLVisionBlock(const Qwen3VLVisionConfiguration& config)
    : attention_(config.hidden_size, config.num_heads),
      mlp_(config.hidden_size, config.intermediate_size),
      norm1_weight_(mx::ones({config.hidden_size})),
      norm1_bias_(mx::zeros({config.hidden_size})),
      norm2_weight_(mx::ones({config.hidden_size})),
      norm2_bias_(mx::zeros({config.hidden_size})),
      eps_(1e-6f)
{}

mx::array Qwen3VLVisionBlock::operator()(
    const mx::array& hidden_states,
    const mx::array& cu_seqlens,
    const mx::array& rotary_pos_emb)
{
    auto h = mx::add(hidden_states,
        attention_(mx::fast::layer_norm(hidden_states, norm1_weight_, norm1_bias_, eps_),
                   cu_seqlens, rotary_pos_emb));
    h = mx::add(h, mlp_(mx::fast::layer_norm(h, norm2_weight_, norm2_bias_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> Qwen3VLVisionBlock::weight_map() {
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

Qwen3VLVisionModel::Qwen3VLVisionModel(const Qwen3VLVisionConfiguration& config)
    : patch_embed_(config.patch_size, config.temporal_patch_size, config.in_channels, config.hidden_size),
      rotary_pos_emb_(config.hidden_size / config.num_heads / 2, 10000.0f),
      pos_embed_weight_(mx::zeros({config.num_position_embeddings, config.hidden_size})),
      merger_(config, false), // main merger: usePostShuffleNorm = false
      deepstack_visual_indexes_(config.deepstack_visual_indexes),
      spatial_merge_size_(config.spatial_merge_size),
      num_grid_per_side_(static_cast<int>(std::sqrt(static_cast<double>(config.num_position_embeddings)))),
      hidden_size_(config.hidden_size),
      in_channels_(config.in_channels)
{
    blocks_.reserve(config.depth);
    for (int i = 0; i < config.depth; ++i)
        blocks_.emplace_back(config);

    // Create deepstack mergers (usePostShuffleNorm = true)
    deepstack_mergers_.reserve(config.deepstack_visual_indexes.size());
    for (size_t i = 0; i < config.deepstack_visual_indexes.size(); ++i)
        deepstack_mergers_.emplace_back(config, true);
}

mx::array Qwen3VLVisionModel::compute_rotary_pos_emb(const std::vector<THW>& grids) {
    int max_hw = 0;
    for (const auto& g : grids) max_hw = std::max(max_hw, std::max(g.h, g.w));
    if (max_hw <= 0) return mx::zeros({1, 1}, mx::float32);

    auto freq_table = rotary_pos_emb_(max_hw);
    int half_dim = freq_table.shape(-1);
    int merge = spatial_merge_size_;

    std::vector<mx::array> all_coords;

    for (const auto& grid : grids) {
        int merged_h = grid.h / merge;
        int merged_w = grid.w / merge;
        if (merged_h <= 0 || merged_w <= 0) continue;

        // Generate block and intra-block indices
        auto block_rows = mx::reshape(mx::astype(mx::arange(0, merged_h), mx::int32),
                                       {merged_h, 1, 1, 1});
        auto block_cols = mx::reshape(mx::astype(mx::arange(0, merged_w), mx::int32),
                                       {1, merged_w, 1, 1});
        auto intra = mx::astype(mx::arange(0, merge), mx::int32);
        auto intra_row = mx::reshape(intra, {1, 1, merge, 1});
        auto intra_col = mx::reshape(intra, {1, 1, 1, merge});

        auto merge_scalar = mx::array(static_cast<int32_t>(merge));
        auto h_index = mx::add(mx::multiply(block_rows, merge_scalar), intra_row);
        auto w_index = mx::add(mx::multiply(block_cols, merge_scalar), intra_col);

        // Broadcast to full shape
        h_index = mx::broadcast_to(h_index, {merged_h, merged_w, merge, merge});
        w_index = mx::broadcast_to(w_index, {merged_h, merged_w, merge, merge});

        auto h_flat = mx::flatten(h_index);
        auto w_flat = mx::flatten(w_index);
        auto coords = mx::stack({h_flat, w_flat}, -1); // [N, 2]

        // Tile for temporal frames
        if (grid.t > 1) {
            std::vector<mx::array> tiled;
            for (int ti = 0; ti < grid.t; ++ti)
                tiled.push_back(coords);
            coords = mx::concatenate(tiled, 0);
        }

        all_coords.push_back(coords);
    }

    if (all_coords.empty()) {
        return mx::zeros({0, half_dim * 2}, freq_table.dtype());
    }

    auto all_coords_concat = mx::concatenate(all_coords, 0); // [total, 2]

    // Extract h and w indices, lookup in frequency table
    auto h_indices = mx::astype(
        mx::slice(all_coords_concat, {0, 0}, {all_coords_concat.shape(0), 1}), mx::int32);
    h_indices = mx::squeeze(h_indices, -1);
    auto w_indices = mx::astype(
        mx::slice(all_coords_concat, {0, 1}, {all_coords_concat.shape(0), 2}), mx::int32);
    w_indices = mx::squeeze(w_indices, -1);

    auto h_embeds = mx::take(freq_table, h_indices, 0);
    auto w_embeds = mx::take(freq_table, w_indices, 0);

    return mx::concatenate({h_embeds, w_embeds}, -1);
}

mx::array Qwen3VLVisionModel::compute_positional_embeddings(const std::vector<THW>& grids) {
    int max_index = num_grid_per_side_ - 1;
    int merge = spatial_merge_size_;

    // Collect bilinear interpolation data for all grids
    std::vector<mx::array> corner_indices(4, mx::array(0.0f));
    std::vector<mx::array> corner_weights(4, mx::array(0.0f));
    std::vector<int> grid_sizes;

    std::vector<std::vector<mx::array>> ci_parts(4);
    std::vector<std::vector<mx::array>> cw_parts(4);

    for (const auto& grid : grids) {
        int h = grid.h;
        int w = grid.w;
        grid_sizes.push_back(h * w);

        // Create linspace indices
        auto h_linspace = mx::astype(mx::arange(0, h), mx::float32);
        h_linspace = mx::divide(
            mx::multiply(h_linspace, mx::array(static_cast<float>(max_index))),
            mx::array(static_cast<float>(std::max(1, h - 1))));

        auto w_linspace = mx::astype(mx::arange(0, w), mx::float32);
        w_linspace = mx::divide(
            mx::multiply(w_linspace, mx::array(static_cast<float>(max_index))),
            mx::array(static_cast<float>(std::max(1, w - 1))));

        // Floor and ceil
        auto h_floor = mx::astype(h_linspace, mx::int32);
        auto h_ceil = mx::minimum(mx::add(h_floor, mx::array(1)), mx::array(max_index));
        auto dh = mx::subtract(h_linspace, mx::astype(h_floor, mx::float32));

        auto w_floor = mx::astype(w_linspace, mx::int32);
        auto w_ceil = mx::minimum(mx::add(w_floor, mx::array(1)), mx::array(max_index));
        auto dw = mx::subtract(w_linspace, mx::astype(w_floor, mx::float32));

        // Broadcast to create meshgrid [h, 1] x [1, w]
        auto h_floor_exp = mx::expand_dims(h_floor, 1);
        auto h_ceil_exp = mx::expand_dims(h_ceil, 1);
        auto w_floor_exp = mx::expand_dims(w_floor, 0);
        auto w_ceil_exp = mx::expand_dims(w_ceil, 0);

        auto base_h = mx::multiply(h_floor_exp, mx::array(num_grid_per_side_));
        auto base_h_ceil = mx::multiply(h_ceil_exp, mx::array(num_grid_per_side_));

        // 4 corner indices (top-left, top-right, bottom-left, bottom-right)
        ci_parts[0].push_back(mx::flatten(mx::add(base_h, w_floor_exp)));
        ci_parts[1].push_back(mx::flatten(mx::add(base_h, w_ceil_exp)));
        ci_parts[2].push_back(mx::flatten(mx::add(base_h_ceil, w_floor_exp)));
        ci_parts[3].push_back(mx::flatten(mx::add(base_h_ceil, w_ceil_exp)));

        // Bilinear weights
        auto dh_exp = mx::expand_dims(dh, 1);
        auto dw_exp = mx::expand_dims(dw, 0);
        auto one_minus_dh = mx::subtract(mx::array(1.0f), dh_exp);
        auto one_minus_dw = mx::subtract(mx::array(1.0f), dw_exp);

        cw_parts[0].push_back(mx::flatten(mx::multiply(one_minus_dh, one_minus_dw)));
        cw_parts[1].push_back(mx::flatten(mx::multiply(one_minus_dh, dw_exp)));
        cw_parts[2].push_back(mx::flatten(mx::multiply(dh_exp, one_minus_dw)));
        cw_parts[3].push_back(mx::flatten(mx::multiply(dh_exp, dw_exp)));
    }

    if (ci_parts[0].empty()) {
        return mx::zeros({0, hidden_size_}, pos_embed_weight_.dtype());
    }

    // Concatenate all corner data
    for (int c = 0; c < 4; ++c) {
        corner_indices[c] = mx::astype(mx::concatenate(ci_parts[c], 0), mx::int32);
        corner_weights[c] = mx::astype(mx::concatenate(cw_parts[c], 0), pos_embed_weight_.dtype());
    }

    // Batch embedding lookup: weighted sum of 4 corners
    int total_patches = corner_indices[0].shape(0);
    auto patch_pos_embeds = mx::zeros({total_patches, hidden_size_}, pos_embed_weight_.dtype());

    for (int c = 0; c < 4; ++c) {
        auto corner_embeds = mx::take(pos_embed_weight_, corner_indices[c], 0);
        auto weighted = mx::multiply(corner_embeds, mx::expand_dims(corner_weights[c], -1));
        patch_pos_embeds = mx::add(patch_pos_embeds, weighted);
    }

    // Split by grid and reshape for spatial merge pattern
    std::vector<mx::array> result_embeds;
    int offset = 0;

    for (size_t gi = 0; gi < grids.size(); ++gi) {
        int h = grids[gi].h;
        int w = grids[gi].w;
        int t = grids[gi].t;
        int size = grid_sizes[gi];

        auto pos_embed = mx::slice(patch_pos_embeds,
                                    {offset, 0},
                                    {offset + size, hidden_size_});
        offset += size;

        int feature_dim = pos_embed.shape(-1);

        // Tile for temporal dimension
        if (t > 1) {
            std::vector<mx::array> temporal_tiles;
            for (int ti = 0; ti < t; ++ti) temporal_tiles.push_back(pos_embed);
            pos_embed = mx::concatenate(temporal_tiles, 0);
        }

        // Reshape for merge pattern: [t, h, w, D] -> [t, h/m, m, w/m, m, D] -> transpose -> flatten
        pos_embed = mx::reshape(pos_embed, {t, h, w, feature_dim});
        pos_embed = mx::reshape(pos_embed, {t, h / merge, merge, w / merge, merge, feature_dim});
        pos_embed = mx::transpose(pos_embed, {0, 1, 3, 2, 4, 5});
        pos_embed = mx::reshape(pos_embed, {-1, feature_dim});

        result_embeds.push_back(pos_embed);
    }

    return mx::concatenate(result_embeds, 0);
}

mx::array Qwen3VLVisionModel::compute_cu_seqlens(const std::vector<THW>& grids) {
    std::vector<int> seq_lengths;
    for (const auto& grid : grids) {
        int per_frame = grid.h * grid.w;
        for (int ti = 0; ti < grid.t; ++ti) {
            seq_lengths.push_back(per_frame);
        }
    }

    if (seq_lengths.empty()) {
        return mx::array({0}, {1}, mx::int32);
    }

    // Compute cumulative sum with leading zero
    std::vector<int32_t> cu_seqlens = {0};
    int running = 0;
    for (int len : seq_lengths) {
        running += len;
        cu_seqlens.push_back(static_cast<int32_t>(running));
    }

    return mx::array(cu_seqlens.data(),
                      {static_cast<int>(cu_seqlens.size())},
                      mx::int32);
}

std::pair<mx::array, std::vector<mx::array>>
Qwen3VLVisionModel::operator()(const mx::array& pixel_values, const std::vector<THW>& grid_thw) {
    auto hidden_states = patch_embed_(pixel_values);

    // Add learned positional embeddings
    auto pos_embeds = compute_positional_embeddings(grid_thw);
    hidden_states = mx::add(hidden_states, pos_embeds);

    auto rotary_embeds = compute_rotary_pos_emb(grid_thw);
    auto cu_seqlens = compute_cu_seqlens(grid_thw);

    std::vector<mx::array> deepstack_outputs;

    for (size_t i = 0; i < blocks_.size(); ++i) {
        hidden_states = blocks_[i](hidden_states, cu_seqlens, rotary_embeds);

        // Check if this block index is a deepstack index
        for (size_t ds = 0; ds < deepstack_visual_indexes_.size(); ++ds) {
            if (static_cast<int>(i) == deepstack_visual_indexes_[ds]) {
                auto feature = deepstack_mergers_[ds](hidden_states);
                deepstack_outputs.push_back(feature);
                break;
            }
        }
    }

    hidden_states = merger_(hidden_states);
    return {hidden_states, deepstack_outputs};
}

std::unordered_map<std::string, mx::array> Qwen3VLVisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("position_ids") != std::string::npos) {
            continue;
        }
        if (k.find("patch_embed.proj.weight") != std::string::npos) {
            // Check if already in MLX format [out, T, H, W, in]
            if (v.ndim() == 5 && v.shape(-1) == in_channels_) {
                sanitized.insert_or_assign(k, v);
            } else if (v.ndim() == 5) {
                // PyTorch format [out, in, T, H, W] -> MLX [out, T, H, W, in]
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

std::unordered_map<std::string, mx::array*> Qwen3VLVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : patch_embed_.weight_map()) map["patch_embed." + k] = v;
    map["pos_embed.weight"] = &pos_embed_weight_;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        auto prefix = "blocks." + std::to_string(i) + ".";
        for (auto& [k, v] : blocks_[i].weight_map()) map[prefix + k] = v;
    }
    for (auto& [k, v] : merger_.weight_map()) map["merger." + k] = v;
    for (size_t i = 0; i < deepstack_mergers_.size(); ++i) {
        auto prefix = "deepstack_merger_list." + std::to_string(i) + ".";
        for (auto& [k, v] : deepstack_mergers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// ── Language Components ────────────────────────────────────────────────

// -- RotaryEmbedding --

Qwen3VLRotaryEmbedding::Qwen3VLRotaryEmbedding(int head_dim, float base,
                                                   const std::optional<Qwen3VLRoPEScaling>& scaling)
    : inv_freq_(mx::divide(
          mx::array(1.0f),
          mx::power(mx::array(base),
                    mx::divide(mx::astype(mx::arange(0, head_dim, 2), mx::float32),
                               mx::array(static_cast<float>(head_dim))))))
{
    if (scaling.has_value()) {
        mrope_section_ = scaling->mrope_section;
    } else {
        mrope_section_ = {24, 20, 20};
    }
}

mx::array Qwen3VLRotaryEmbedding::apply_interleaved_mrope(const mx::array& freqs) {
    // freqs: [3, B, L, head_dim/2]
    // Select dimension 0 (temporal) as base, then interleave with dim 1 (height) and dim 2 (width)
    // based on mrope_section

    // Extract the temporal (base) frequencies: freqs[0, :, :, :]
    auto freqs_t = mx::slice(freqs, {0, 0, 0, 0},
                              {1, freqs.shape(1), freqs.shape(2), freqs.shape(3)});
    freqs_t = mx::squeeze(freqs_t, 0); // [B, L, head_dim/2]

    int dims = freqs_t.shape(-1);
    int batch = freqs_t.shape(0);
    int seq_len = freqs_t.shape(1);

    // Build the interleaved result
    // For each frequency index idx:
    //   - If idx matches dim 1 pattern: use freqs[1, :, :, idx]
    //   - If idx matches dim 2 pattern: use freqs[2, :, :, idx]
    //   - Otherwise: use freqs[0, :, :, idx] (temporal)

    // Precompute which dimension each index belongs to
    std::vector<int> dim_assignment(dims, 0); // 0 = temporal (default)

    for (int idx = 0; idx < dims; ++idx) {
        // Check dimension 1 (height): offset=1, section boundary = mrope_section[1] * 3
        int end1 = std::min(mrope_section_[1] * 3, dims);
        if (idx >= 1 && idx < end1 && (idx - 1) % 3 == 0) {
            dim_assignment[idx] = 1;
            continue;
        }
        // Check dimension 2 (width): offset=2, section boundary = mrope_section[2] * 3
        int end2 = std::min(mrope_section_[2] * 3, dims);
        if (idx >= 2 && idx < end2 && (idx - 2) % 3 == 0) {
            dim_assignment[idx] = 2;
            continue;
        }
    }

    // Gather slices based on dim_assignment
    std::vector<mx::array> slices;
    slices.reserve(dims);

    for (int idx = 0; idx < dims; ++idx) {
        int d = dim_assignment[idx];
        // freqs[d, :, :, idx]
        auto slice = mx::slice(freqs,
                                {d, 0, 0, idx},
                                {d + 1, batch, seq_len, idx + 1});
        slice = mx::squeeze(slice, 0);  // [B, L, 1]
        slice = mx::squeeze(slice, -1); // [B, L]
        slices.push_back(slice);
    }

    return mx::stack(slices, -1); // [B, L, dims]
}

std::pair<mx::array, mx::array>
Qwen3VLRotaryEmbedding::operator()(const mx::array& position_ids, mx::Dtype dtype) {
    // position_ids: [3, B, L] or [B, L]
    auto pos_ids = position_ids;
    if (pos_ids.ndim() == 2) {
        // Expand to [1, B, L] then tile to [3, B, L]
        pos_ids = mx::expand_dims(pos_ids, 0);
        std::vector<mx::array> tiles;
        for (int i = 0; i < 3; ++i) tiles.push_back(pos_ids);
        pos_ids = mx::concatenate(tiles, 0);
    }

    auto pos = mx::astype(pos_ids, mx::float32); // [3, B, L]
    auto inv_f = mx::astype(inv_freq_, mx::float32);
    // inv_f: [D/2] -> [1, 1, 1, D/2]
    inv_f = mx::reshape(inv_f, {1, 1, 1, inv_f.shape(0)});

    // pos: [3, B, L] -> [3, B, L, 1]
    auto pos_exp = mx::expand_dims(pos, -1);
    auto freqs = mx::multiply(pos_exp, inv_f); // [3, B, L, D/2]

    // Apply interleaved MRoPE
    freqs = apply_interleaved_mrope(freqs); // [B, L, D/2]

    // Duplicate for full head_dim: [B, L, D]
    auto emb = mx::concatenate({freqs, freqs}, -1);
    auto cos_vals = mx::astype(mx::cos(emb), dtype);
    auto sin_vals = mx::astype(mx::sin(emb), dtype);

    return {cos_vals, sin_vals};
}

// -- Language Attention --

Qwen3VLLanguageAttention::Qwen3VLLanguageAttention(const Qwen3VLTextConfiguration& config)
    : heads_(config.num_attention_heads),
      kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      q_norm_weight_(mx::ones({config.head_dim})),
      k_norm_weight_(mx::ones({config.head_dim})),
      rms_norm_eps_(config.rms_norm_eps),
      rotary_emb_(config.head_dim, config.rope_theta, config.rope_scaling)
{}

mx::array Qwen3VLLanguageAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache,
    const std::optional<mx::array>& position_ids)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Reshape and apply Q/K norms
    queries = mx::reshape(queries, {B, L, heads_, head_dim_});
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3}); // [B, H, L, D]

    keys = mx::reshape(keys, {B, L, kv_heads_, head_dim_});
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::reshape(values, {B, L, kv_heads_, head_dim_});
    values = mx::transpose(values, {0, 2, 1, 3});

    // Compute position IDs if not provided
    mx::array pos_ids = mx::array(0.0f);
    int kv_seq_len = keys.shape(2);

    if (position_ids.has_value()) {
        pos_ids = position_ids.value();
        if (cache) {
            kv_seq_len += cache->offset() + 1;
        }
    } else {
        int offset = cache ? cache->offset() : 0;
        kv_seq_len += offset + 1;
        // Create default position IDs: [3, B, L]
        auto base = mx::astype(mx::arange(offset, offset + L), mx::int32);
        base = mx::reshape(base, {1, L});
        base = mx::broadcast_to(base, {B, L});
        pos_ids = mx::expand_dims(base, 0);
        // Tile to [3, B, L]
        std::vector<mx::array> tiles;
        for (int i = 0; i < 3; ++i) tiles.push_back(pos_ids);
        pos_ids = mx::concatenate(tiles, 0);
    }

    // Apply rotary embedding
    auto [cos_vals, sin_vals] = rotary_emb_(pos_ids, x.dtype());
    auto [q_embed, k_embed] = apply_multimodal_rotary(queries, keys, cos_vals, sin_vals);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(k_embed, values);
        k_embed = k;
        values = v;
    }

    // Compute attention mask — slice array mask if needed
    AttentionMask attn_mask = mask;
    if (mask.has_array()) {
        auto m = mask.as_array();
        if (m.shape(-1) > kv_seq_len) {
            mx::Shape start_s(m.ndim(), 0);
            mx::Shape stop_s(m.shape());
            stop_s.back() = kv_seq_len;
            m = mx::slice(m, start_s, stop_s);
        }
        attn_mask = AttentionMask::from_array(m);
    }

    auto output = sdpa(
        q_embed, k_embed, values, scale_, attn_mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3VLLanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// -- Language MLP --

Qwen3VLLanguageMLP::Qwen3VLLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen3VLLanguageMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3VLLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Decoder Layer --

Qwen3VLDecoderLayer::Qwen3VLDecoderLayer(const Qwen3VLTextConfiguration& config)
    : attention_(config),
      mlp_(config.hidden_size, config.intermediate_size),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Qwen3VLDecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache,
    const std::optional<mx::array>& position_ids)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_),
                        mask, cache, position_ids);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen3VLDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

Qwen3VLLanguageModelInner::Qwen3VLLanguageModelInner(const Qwen3VLTextConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array Qwen3VLLanguageModelInner::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding,
    const AttentionMask& mask,
    const std::optional<mx::array>& position_ids,
    const std::optional<mx::array>& visual_mask,
    const std::vector<mx::array>* deepstack_embeds)
{
    mx::array h = mx::array(0.0f);
    if (input_embedding.has_value()) {
        h = input_embedding.value();
    } else if (inputs.has_value()) {
        h = mx::take(embed_tokens_weight_, inputs.value(), 0);
    } else {
        throw std::runtime_error("Either inputs or input_embedding must be provided");
    }

    auto attn_mask = mask;
    if (attn_mask.is_none()) {
        attn_mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, attn_mask, lc, position_ids);

        // Apply deepstack: add visual embeddings at visual token positions
        if (deepstack_embeds && i < deepstack_embeds->size() && visual_mask.has_value()) {
            // Find indices where visual_mask is true
            auto vm = visual_mask.value();
            mx::eval(vm);

            auto vm_bool = mx::astype(vm, mx::bool_);
            mx::eval(vm_bool);

            // Get the number of visual tokens
            auto vm_flat = mx::flatten(vm_bool);
            mx::eval(vm_flat);

            int num_tokens = vm_flat.shape(0);
            auto vm_data = vm_flat.data<bool>();

            std::vector<uint32_t> indices;
            for (int idx = 0; idx < num_tokens; ++idx) {
                if (vm_data[idx]) {
                    indices.push_back(static_cast<uint32_t>(idx));
                }
            }

            if (!indices.empty()) {
                auto index_arr = mx::array(indices.data(),
                                            {static_cast<int>(indices.size())},
                                            mx::uint32);

                // h[:, indices, :] += deepstack_embeds[i]
                auto visual_embeds = (*deepstack_embeds)[i];

                // Gather existing values at visual positions
                auto existing = mx::take(mx::squeeze(h, 0), index_arr, 0); // [N, D]
                auto updated = mx::add(existing, visual_embeds); // [N, D]

                // Scatter back using slice_update on the flattened sequence
                auto h_squeezed = mx::squeeze(h, 0); // [L, D]
                for (size_t vi = 0; vi < indices.size(); ++vi) {
                    int idx = static_cast<int>(indices[vi]);
                    auto val = mx::slice(updated,
                                          {static_cast<int>(vi), 0},
                                          {static_cast<int>(vi) + 1, h.shape(-1)});
                    h_squeezed = mx::slice_update(h_squeezed, val,
                                                   mx::Shape{idx, 0},
                                                   mx::Shape{idx + 1, h.shape(-1)});
                }
                h = mx::expand_dims(h_squeezed, 0);
            }
        }
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen3VLLanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array Qwen3VLLanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen3VLLanguageModelInner::weight_map() {
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

Qwen3VLLanguageModel::Qwen3VLLanguageModel(const Qwen3VLConfiguration& config)
    : model_(config.text_config),
      config_(config)
{
    kv_heads_.resize(config.text_config.num_hidden_layers, config.text_config.num_key_value_heads);
    if (!config.text_config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.text_config.vocab_size, config.text_config.hidden_size});
    }
}

LMOutput Qwen3VLLanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding,
    const AttentionMask& mask,
    const std::optional<mx::array>& position_ids_in,
    const std::optional<mx::array>& visual_mask,
    const std::vector<mx::array>* deepstack_embeds,
    const mx::array* pixel_values,
    const std::vector<THW>* image_grid_thw,
    const std::vector<THW>* video_grid_thw)
{
    // Reset rope deltas when new pixel values arrive
    if (pixel_values) {
        rope_deltas_ = std::nullopt;
    }

    auto position_ids = position_ids_in;

    // Compute position IDs if not provided
    if (!position_ids.has_value() && (!mask.has_array() || mask.as_array().ndim() == 2)) {
        int cache_offset = (cache && !cache->empty()) ? (*cache)[0].offset() : 0;

        if (cache_offset == 0 || !rope_deltas_.has_value() || !cache) {
            // Initial prefill or no cache
            if (inputs.has_value()) {
                auto [computed_pos, deltas] = get_rope_index(
                    inputs.value(),
                    image_grid_thw, video_grid_thw,
                    config_.vision_config.spatial_merge_size,
                    config_.base_config.image_token_id,
                    config_.base_config.video_token_id,
                    config_.base_config.vision_start_token_id,
                    mask.has_array() ? &mask.as_array() : nullptr);
                position_ids = computed_pos;
                rope_deltas_ = deltas;
            } else if (cache && !rope_deltas_.has_value()) {
                // Embeddings provided, no rope deltas yet
                int batch = input_embedding->shape(0);
                int seq_len = input_embedding->shape(1);
                int current_offset = (cache && !cache->empty()) ? (*cache)[0].offset() : 0;

                auto base = mx::astype(mx::arange(0, seq_len), mx::int32);
                base = mx::reshape(base, {1, seq_len});
                base = mx::broadcast_to(base, {batch, seq_len});
                base = mx::add(base, mx::array(static_cast<int32_t>(current_offset)));

                auto pos3d = mx::expand_dims(base, 0);
                // Broadcast to [3, batch, seq_len]
                pos3d = mx::broadcast_to(pos3d, {3, batch, seq_len});
                position_ids = pos3d;
            }
        } else if (cache && rope_deltas_.has_value()) {
            // Subsequent generation steps
            auto& input_ref = inputs.has_value() ? inputs.value() : input_embedding.value();
            int batch = input_ref.shape(0);
            int seq_len = input_ref.shape(1);

            int last_cache_offset = cache->back().offset();
            auto delta = mx::add(
                mx::array(static_cast<int32_t>(last_cache_offset)),
                mx::astype(rope_deltas_.value(), mx::int32));

            auto base = mx::astype(mx::arange(0, seq_len), mx::int32);
            base = mx::reshape(base, {1, seq_len});
            base = mx::broadcast_to(base, {batch, seq_len});

            if (delta.shape(0) == 1 && batch > 1) {
                // Broadcast delta
                std::vector<mx::array> delta_tiles;
                for (int i = 0; i < batch; ++i) delta_tiles.push_back(delta);
                delta = mx::concatenate(delta_tiles, 0);
            }

            base = mx::add(base, delta);
            auto pos3d = mx::expand_dims(base, 0);
            pos3d = mx::broadcast_to(pos3d, {3, batch, seq_len});
            position_ids = pos3d;
        }
    }

    auto out = model_(inputs, cache, input_embedding, AttentionMask{} /*mask*/,
                       position_ids, visual_mask, deepstack_embeds);

    if (lm_head_weight_.has_value()) {
        out = mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    } else {
        out = model_.embed_as_linear(out);
    }

    return LMOutput(out);
}

// Static method: compute MRoPE position IDs
std::pair<mx::array, mx::array> Qwen3VLLanguageModel::get_rope_index(
    const mx::array& input_ids,
    const std::vector<THW>* image_grid_thw,
    const std::vector<THW>* video_grid_thw,
    int spatial_merge_size,
    int image_token_id,
    int video_token_id,
    int vision_start_token_id,
    const mx::array* attention_mask)
{
    int batch_size = input_ids.shape(0);
    int seq_length = input_ids.shape(1);

    auto position_ids = mx::astype(mx::arange(0, seq_length), mx::int32);
    position_ids = mx::broadcast_to(
        mx::reshape(position_ids, {1, seq_length}),
        {batch_size, seq_length});

    // If no vision inputs, return simple 3D position IDs
    if (!image_grid_thw && !video_grid_thw) {
        auto pos3d = mx::broadcast_to(
            mx::expand_dims(position_ids, 0),
            {3, batch_size, seq_length});
        auto zeros = mx::zeros({batch_size}, mx::int32);
        return {pos3d, zeros};
    }

    // Initialize position_ids for multimodal case
    position_ids = mx::astype(mx::ones({batch_size, seq_length}), mx::int32);
    position_ids = mx::broadcast_to(
        mx::expand_dims(position_ids, 0),
        {3, batch_size, seq_length});

    std::vector<int> mrope_position_deltas;

    // Get mask (default to ones)
    mx::array mask_arr = mx::ones({batch_size, seq_length}, mx::int32);
    if (attention_mask) {
        mask_arr = *attention_mask;
    } else {
        mask_arr = mx::ones({batch_size, seq_length}, mx::int32);
    }

    // Process each batch item
    for (int bi = 0; bi < batch_size; ++bi) {
        // Extract batch input IDs
        auto batch_ids = mx::slice(input_ids, {bi, 0}, {bi + 1, seq_length});
        batch_ids = mx::squeeze(batch_ids, 0); // [L]

        // Mask out padding
        auto batch_mask = mx::slice(mask_arr, {bi, 0}, {bi + 1, seq_length});
        batch_mask = mx::squeeze(batch_mask, 0);
        batch_ids = mx::where(mx::equal(batch_mask, mx::array(1)), batch_ids, mx::zeros_like(batch_ids));
        mx::eval(batch_ids);

        // Count image and video tokens
        auto img_count_arr = mx::sum(mx::astype(mx::equal(batch_ids, mx::array(image_token_id)), mx::int32));
        auto vid_count_arr = mx::sum(mx::astype(mx::equal(batch_ids, mx::array(video_token_id)), mx::int32));
        mx::eval(img_count_arr);
        mx::eval(vid_count_arr);
        int image_nums = img_count_arr.item<int>();
        int video_nums = vid_count_arr.item<int>();

        // Read token IDs to CPU
        auto tokens_data = batch_ids.data<int32_t>();
        std::vector<int> input_tokens(tokens_data, tokens_data + seq_length);

        std::vector<mx::array> llm_pos_ids_list;
        int st = 0;
        int remain_images = image_nums;
        int remain_videos = video_nums;
        int image_index = 0;
        int video_index = 0;

        for (int step = 0; step < (image_nums + video_nums); ++step) {
            // Find next image token
            int ed_image = seq_length + 1;
            if (remain_images > 0) {
                for (int idx = st; idx < seq_length; ++idx) {
                    if (input_tokens[idx] == image_token_id) {
                        ed_image = idx;
                        break;
                    }
                }
            }

            // Find next video token
            int ed_video = seq_length + 1;
            if (remain_videos > 0) {
                for (int idx = st; idx < seq_length; ++idx) {
                    if (input_tokens[idx] == video_token_id) {
                        ed_video = idx;
                        break;
                    }
                }
            }

            int t, h, w, ed;
            if (ed_image < ed_video) {
                if (!image_grid_thw || image_index >= static_cast<int>(image_grid_thw->size())) break;
                t = (*image_grid_thw)[image_index].t;
                h = (*image_grid_thw)[image_index].h;
                w = (*image_grid_thw)[image_index].w;
                image_index++;
                remain_images--;
                ed = ed_image;
            } else {
                if (!video_grid_thw || video_index >= static_cast<int>(video_grid_thw->size())) break;
                t = (*video_grid_thw)[video_index].t;
                h = (*video_grid_thw)[video_index].h;
                w = (*video_grid_thw)[video_index].w;
                video_index++;
                remain_videos--;
                ed = ed_video;
            }

            int llm_grid_t = t;
            int llm_grid_h = h / spatial_merge_size;
            int llm_grid_w = w / spatial_merge_size;

            // Get starting index
            int st_idx = 0;
            if (!llm_pos_ids_list.empty()) {
                auto last = llm_pos_ids_list.back();
                auto max_val = mx::max(last);
                mx::eval(max_val);
                st_idx = max_val.item<int>() + 1;
            }

            // Add text tokens before this visual block
            int text_len = ed - st;
            if (text_len > 0) {
                auto index = mx::astype(mx::arange(0, text_len), mx::int32);
                index = mx::reshape(index, {1, text_len});
                index = mx::broadcast_to(index, {3, text_len});
                index = mx::add(index, mx::array(static_cast<int32_t>(st_idx)));
                llm_pos_ids_list.push_back(index);
            }

            // Add 3D position IDs for visual tokens
            auto t_index = mx::astype(mx::arange(0, llm_grid_t), mx::int32);
            t_index = mx::reshape(t_index, {llm_grid_t, 1});
            t_index = mx::broadcast_to(t_index, {llm_grid_t, llm_grid_h * llm_grid_w});
            t_index = mx::flatten(t_index);

            auto h_index = mx::astype(mx::arange(0, llm_grid_h), mx::int32);
            h_index = mx::reshape(h_index, {1, llm_grid_h, 1});
            h_index = mx::broadcast_to(h_index, {llm_grid_t, llm_grid_h, llm_grid_w});
            h_index = mx::flatten(h_index);

            auto w_index = mx::astype(mx::arange(0, llm_grid_w), mx::int32);
            w_index = mx::reshape(w_index, {1, 1, llm_grid_w});
            w_index = mx::broadcast_to(w_index, {llm_grid_t, llm_grid_h, llm_grid_w});
            w_index = mx::flatten(w_index);

            auto visual_pos_ids = mx::stack({t_index, h_index, w_index}, 0); // [3, N]
            visual_pos_ids = mx::add(visual_pos_ids,
                                      mx::array(static_cast<int32_t>(text_len + st_idx)));
            llm_pos_ids_list.push_back(visual_pos_ids);

            st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
        }

        // Add remaining text tokens
        if (st < seq_length) {
            int st_idx = 0;
            if (!llm_pos_ids_list.empty()) {
                auto last = llm_pos_ids_list.back();
                auto max_val = mx::max(last);
                mx::eval(max_val);
                st_idx = max_val.item<int>() + 1;
            }

            int text_len = seq_length - st;
            auto t_index = mx::astype(mx::arange(0, text_len), mx::int32);
            t_index = mx::reshape(t_index, {1, text_len});
            t_index = mx::broadcast_to(t_index, {3, text_len});
            t_index = mx::add(t_index, mx::array(static_cast<int32_t>(st_idx)));
            llm_pos_ids_list.push_back(t_index);
        }

        // Concatenate all position IDs for this batch item
        if (!llm_pos_ids_list.empty()) {
            auto llm_positions = mx::concatenate(llm_pos_ids_list, 1); // [3, total_seq]

            // Compute max position for rope delta
            auto max_pos = mx::max(llm_positions);
            mx::eval(max_pos);
            int max_pos_id = max_pos.item<int>();
            mrope_position_deltas.push_back(max_pos_id + 1 - seq_length);

            // Update position_ids for this batch
            // For batch size = 1 (most common case)
            auto expanded_positions = mx::expand_dims(llm_positions, 1); // [3, 1, L]

            auto batch_mask_expanded = mx::broadcast_to(
                mx::reshape(mx::slice(mask_arr, {bi, 0}, {bi + 1, seq_length}),
                             {1, 1, seq_length}),
                {3, 1, seq_length});

            auto current_pos = mx::slice(position_ids,
                                          {0, bi, 0}, {3, bi + 1, seq_length});

            auto new_pos = mx::where(batch_mask_expanded, expanded_positions, current_pos);
            position_ids = new_pos;
        }
    }

    // Return deltas
    mx::array deltas = mx::zeros({batch_size}, mx::int32);
    if (mrope_position_deltas.empty()) {
        deltas = mx::zeros({batch_size}, mx::int32);
    } else {
        std::vector<int32_t> delta_data;
        for (int d : mrope_position_deltas) delta_data.push_back(static_cast<int32_t>(d));
        deltas = mx::array(delta_data.data(), {static_cast<int>(delta_data.size())}, mx::int32);
    }

    return {position_ids, deltas};
}

std::unordered_map<std::string, mx::array*> Qwen3VLLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Top-Level Model ────────────────────────────────────────────────────

Qwen3VLModel::Qwen3VLModel(const Qwen3VLConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

std::pair<mx::array, mx::array> Qwen3VLModel::merge_input_ids_with_image_features(
    const mx::array& image_features,
    const mx::array& input_embeds,
    const mx::array& input_ids,
    int image_token_index,
    int video_token_index)
{
    // Create mask for image and video tokens
    auto image_mask = mx::equal(input_ids, mx::array(image_token_index));
    auto video_mask = mx::equal(input_ids, mx::array(video_token_index));
    auto special_mask = mx::logical_or(image_mask, video_mask);

    // Expand mask for embedding dimension
    auto special_mask_expanded = mx::expand_dims(special_mask, -1);
    auto mask_expanded = mx::broadcast_to(special_mask_expanded,
                                           {input_embeds.shape(0), input_embeds.shape(1), input_embeds.shape(2)});

    // Flatten everything for scatter
    auto flat_embeds = mx::flatten(input_embeds);
    auto flat_features = mx::flatten(image_features);
    auto flat_mask = mx::flatten(mask_expanded);

    mx::eval(flat_mask);

    // Find non-zero indices
    auto bool_mask = mx::astype(flat_mask, mx::bool_);
    mx::eval(bool_mask);

    auto mask_data = bool_mask.data<bool>();
    int mask_size = bool_mask.shape(0);

    std::vector<uint32_t> nonzero_indices;
    for (int i = 0; i < mask_size; ++i) {
        if (mask_data[i]) nonzero_indices.push_back(static_cast<uint32_t>(i));
    }

    auto result = flat_embeds;
    if (!nonzero_indices.empty() &&
        static_cast<int>(nonzero_indices.size()) == flat_features.shape(0)) {
        auto index_arr = mx::array(nonzero_indices.data(),
                                    {static_cast<int>(nonzero_indices.size())},
                                    mx::uint32);
        // Scatter features into result at nonzero positions
        for (size_t i = 0; i < nonzero_indices.size(); ++i) {
            int idx = static_cast<int>(nonzero_indices[i]);
            auto feat_val = mx::slice(flat_features,
                                       {static_cast<int>(i)},
                                       {static_cast<int>(i) + 1});
            result = mx::slice_update(result, feat_val, mx::Shape{idx}, mx::Shape{idx + 1});
        }
    }

    result = mx::reshape(result, input_embeds.shape());

    // Create visual mask (squeezed, boolean)
    auto visual_mask = mx::astype(mx::squeeze(special_mask_expanded, -1), mx::bool_);

    return {result, visual_mask};
}

PrepareResult Qwen3VLModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;

    const mx::array* pixel_values = nullptr;
    std::vector<THW> image_frames;
    std::vector<THW> video_frames;
    mx::array combined_pixels = mx::array(0.0f);

    std::vector<mx::array> pixel_parts;

    if (input.image.has_value()) {
        pixel_parts.push_back(input.image->pixels);
        if (input.image->frames.has_value()) {
            image_frames.insert(image_frames.end(),
                input.image->frames->begin(), input.image->frames->end());
        }
    }

    if (input.video.has_value()) {
        pixel_parts.push_back(input.video->pixels);
        if (input.video->frames.has_value()) {
            video_frames.insert(video_frames.end(),
                input.video->frames->begin(), input.video->frames->end());
        }
    }

    if (!pixel_parts.empty()) {
        if (pixel_parts.size() == 1) {
            combined_pixels = pixel_parts[0];
        } else {
            combined_pixels = mx::concatenate(pixel_parts, 0);
        }
        pixel_values = &combined_pixels;
    }

    // Combine all frames for vision model
    std::vector<THW> all_frames;
    all_frames.insert(all_frames.end(), image_frames.begin(), image_frames.end());
    all_frames.insert(all_frames.end(), video_frames.begin(), video_frames.end());

    std::optional<mx::array> input_embeddings;
    std::optional<mx::array> visual_mask;
    std::vector<mx::array> deepstack_outputs_storage;
    const std::vector<mx::array>* deepstack_embeds = nullptr;

    if (pixel_values && !all_frames.empty()) {
        // Get text embeddings
        auto text_embeds = language_model_.inner().embed_tokens(input_ids);

        // Run vision model
        auto [vision_hidden, ds_outputs] = vision_tower_(*pixel_values, all_frames);

        // Split vision features by frame and compute merge sizes
        int merge_size = config_.vision_config.spatial_merge_size;
        std::vector<int> splits;
        for (const auto& f : all_frames) {
            splits.push_back(f.product() / (merge_size * merge_size));
        }

        // Compute cumulative split indices
        std::vector<int> split_indices;
        int running = 0;
        for (size_t i = 0; i + 1 < splits.size(); ++i) {
            running += splits[i];
            split_indices.push_back(running);
        }

        // Split and concatenate (preserves order)
        std::vector<mx::array> feature_slices;
        int start = 0;
        for (size_t i = 0; i < splits.size(); ++i) {
            auto sl = mx::slice(vision_hidden, {start, 0},
                                 {start + splits[i], vision_hidden.shape(-1)});
            feature_slices.push_back(sl);
            start += splits[i];
        }
        auto flattened_features = mx::concatenate(feature_slices, 0);
        flattened_features = mx::astype(flattened_features, text_embeds.dtype());

        // Merge vision features into text embeddings
        auto [merged, vmask] = merge_input_ids_with_image_features(
            flattened_features, text_embeds, input_ids,
            config_.base_config.image_token_id,
            config_.base_config.video_token_id);

        input_embeddings = merged;
        visual_mask = vmask;

        // Process deepstack outputs
        if (!ds_outputs.empty()) {
            for (auto& layer_features : ds_outputs) {
                std::vector<mx::array> ds_slices;
                int ds_start = 0;
                for (size_t i = 0; i < splits.size(); ++i) {
                    auto sl = mx::slice(layer_features, {ds_start, 0},
                                         {ds_start + splits[i], layer_features.shape(-1)});
                    ds_slices.push_back(sl);
                    ds_start += splits[i];
                }
                auto concat_slices = mx::concatenate(ds_slices, 0);
                deepstack_outputs_storage.push_back(mx::astype(concat_slices, text_embeds.dtype()));
            }
            deepstack_embeds = &deepstack_outputs_storage;
        }
    }

    // Run language model
    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;

    auto result = language_model_(
        input_ids,
        cache_ptr,
        input_embeddings,
        AttentionMask{}, // mask
        std::nullopt, // position_ids
        visual_mask,
        deepstack_embeds,
        pixel_values,
        image_frames.empty() ? nullptr : &image_frames,
        video_frames.empty() ? nullptr : &video_frames);

    return PrepareResult::logits(std::move(result));
}

LMOutput Qwen3VLModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Qwen3VLModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array> Qwen3VLModel::sanitize_impl(
    std::unordered_map<std::string, mx::array> weights)
{
    // Rename keys: model.visual -> vision_tower, model.language_model -> language_model.model
    std::unordered_map<std::string, mx::array> renamed;

    for (auto& [k, v] : weights) {
        std::string key = k;

        if (key.find("model") != std::string::npos) {
            if (key.find("model.visual") != std::string::npos) {
                size_t pos = key.find("model.visual");
                key.replace(pos, 12, "vision_tower");
            } else if (key.find("model.language_model") != std::string::npos) {
                size_t pos = key.find("model.language_model");
                key.replace(pos, 20, "language_model.model");
            }
        } else if (key.find("lm_head") != std::string::npos) {
            size_t pos = key.find("lm_head");
            key = "language_model.lm_head" + key.substr(pos + 7);
        }

        // Skip lm_head if tie_word_embeddings
        if (config_.text_config.tie_word_embeddings && key.find(".lm_head.") != std::string::npos) {
            continue;
        }

        renamed.insert_or_assign(key, v);
    }

    // Sanitize vision weights (conv format conversion)
    return vision_tower_.sanitize(std::move(renamed));
}

void Qwen3VLModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen3VLModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_tower_.weight_map()) map["vision_tower." + k] = v;
    for (auto& [k, v] : language_model_.weight_map()) map["language_model." + k] = v;
    return map;
}

} // namespace mlx_lm

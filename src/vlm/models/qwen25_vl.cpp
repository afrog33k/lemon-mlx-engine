// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++

#include <mlx-lm/vlm/models/qwen25_vl.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Qwen25VLTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("qwen2_5_vl"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.max_position_embeddings = j.value("max_position_embeddings", 128000);
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);
    c.sliding_window = j.value("sliding_window", 32768);
    c.use_sliding_window = j.value("use_sliding_window", false);

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

void from_json(const nlohmann::json& j, Qwen25VLVisionConfiguration& c) {
    c.depth = j.at("depth").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.out_hidden_size = j.at("out_hidden_size").get<int>();
    c.num_heads = j.at("num_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.in_channels = j.value("in_chans", 3);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
    c.spatial_patch_size = j.at("spatial_patch_size").get<int>();
    c.spatial_merge_size = j.at("spatial_merge_size").get<int>();
    c.temporal_patch_size = j.at("temporal_patch_size").get<int>();
    c.window_size = j.at("window_size").get<int>();
    c.fullatt_block_indexes = j.at("fullatt_block_indexes").get<std::vector<int>>();
    c.tokens_per_second = j.value("tokens_per_second", 0);
    c.skip_vision = j.value("skip_vision", false);
    c.hidden_act = j.value("hidden_act", std::string("silu"));
}

void from_json(const nlohmann::json& j, Qwen25VLBaseConfiguration& c) {
    c.model_type = j.at("model_type").get<std::string>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.image_token_id = j.at("image_token_id").get<int>();
    c.video_token_id = j.at("video_token_id").get<int>();
    c.vision_start_token_id = j.value("vision_start_token_id", 0);
    c.vision_end_token_id = j.value("vision_end_token_id", 0);
    c.vision_token_id = j.value("vision_token_id", 0);
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.sliding_window = j.value("sliding_window", 32768);
    c.use_sliding_window = j.value("use_sliding_window", false);
    c.max_window_layers = j.value("max_window_layers", 0);
}

void from_json(const nlohmann::json& j, Qwen25VLConfiguration& c) {
    // Vision config is a sub-dictionary
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Qwen25VLVisionConfiguration>();
    }
    // Text and base configs are overlaid in the top level
    c.text_config = j.get<Qwen25VLTextConfiguration>();
    c.base_config = j.get<Qwen25VLBaseConfiguration>();
}

void from_json(const nlohmann::json& j, Qwen25VLProcessorConfiguration& c) {
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

// -- PatchMerger (uses RMSNorm instead of Qwen2VL's LayerNorm) --

Qwen25VLPatchMerger::Qwen25VLPatchMerger(int dimensions, int context_dimensions, int spatial_merge_size)
    : hidden_size_(context_dimensions * spatial_merge_size * spatial_merge_size),
      ln_q_weight_(mx::ones({context_dimensions})),
      mlp_0_weight_(mx::zeros({hidden_size_, hidden_size_})),
      mlp_0_bias_(mx::zeros({hidden_size_})),
      mlp_2_weight_(mx::zeros({dimensions, hidden_size_})),
      mlp_2_bias_(mx::zeros({dimensions})),
      eps_(1e-6f)
{}

mx::array Qwen25VLPatchMerger::operator()(const mx::array& x) {
    // RMSNorm then reshape to merge spatial patches
    auto normed = mx::fast::rms_norm(x, ln_q_weight_, eps_);
    auto merged = mx::reshape(normed, {-1, hidden_size_});
    // MLP: Linear -> GELU -> Linear
    merged = linear_fwd(merged, mlp_0_weight_, &mlp_0_bias_);
    merged = gelu(merged);
    merged = linear_fwd(merged, mlp_2_weight_, &mlp_2_bias_);
    return merged;
}

std::unordered_map<std::string, mx::array*> Qwen25VLPatchMerger::weight_map() {
    return {
        {"ln_q.weight", &ln_q_weight_},
        {"mlp.0.weight", &mlp_0_weight_}, {"mlp.0.bias", &mlp_0_bias_},
        {"mlp.2.weight", &mlp_2_weight_}, {"mlp.2.bias", &mlp_2_bias_},
    };
}

// -- Vision Attention (takes attention mask for window/full attention) --

Qwen25VLVisionAttention::Qwen25VLVisionAttention(int dims, int num_heads)
    : num_heads_(num_heads),
      scale_(std::pow(static_cast<float>(dims / num_heads), -0.5f)),
      qkv_weight_(mx::zeros({3 * dims, dims})),
      qkv_bias_(mx::zeros({3 * dims})),
      proj_weight_(mx::zeros({dims, dims})),
      proj_bias_(mx::zeros({dims}))
{}

mx::array Qwen25VLVisionAttention::operator()(
    const mx::array& x,
    const mx::array& attention_mask,
    const mx::array& rotary_pos_emb)
{
    int seq_len = x.shape(0);

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

    // Reshape for attention: [1, num_heads, seq_len, head_dim]
    q = mx::transpose(mx::reshape(q, {1, seq_len, num_heads_, head_dim}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {1, seq_len, num_heads_, head_dim}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {1, seq_len, num_heads_, head_dim}), {0, 2, 1, 3});

    // Use scaled_dot_product_attention without mask (mask is applied via reindexing)
    auto output = mx::fast::scaled_dot_product_attention(q, k, v, scale_);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {seq_len, -1});

    return linear_fwd(output, proj_weight_, &proj_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen25VLVisionAttention::weight_map() {
    return {
        {"qkv.weight", &qkv_weight_}, {"qkv.bias", &qkv_bias_},
        {"proj.weight", &proj_weight_}, {"proj.bias", &proj_bias_},
    };
}

// -- Vision MLP (SiLU gate/up/down with bias) --

Qwen25VLVisionMLP::Qwen25VLVisionMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      gate_bias_(mx::zeros({hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions})),
      up_bias_(mx::zeros({hidden_dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      down_bias_(mx::zeros({dimensions}))
{}

mx::array Qwen25VLVisionMLP::operator()(const mx::array& x) {
    return linear_fwd(
        swiglu(linear_fwd(x, gate_weight_, &gate_bias_),
               linear_fwd(x, up_weight_, &up_bias_)),
        down_weight_, &down_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen25VLVisionMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_}, {"gate_proj.bias", &gate_bias_},
        {"up_proj.weight", &up_weight_}, {"up_proj.bias", &up_bias_},
        {"down_proj.weight", &down_weight_}, {"down_proj.bias", &down_bias_},
    };
}

// -- Vision Block (uses RMSNorm instead of Qwen2VL's LayerNorm) --

Qwen25VLVisionBlock::Qwen25VLVisionBlock(const Qwen25VLVisionConfiguration& config)
    : attention_(config.hidden_size, config.num_heads),
      mlp_(config.hidden_size, config.intermediate_size),
      norm1_weight_(mx::ones({config.hidden_size})),
      norm2_weight_(mx::ones({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array Qwen25VLVisionBlock::operator()(
    const mx::array& hidden_states,
    const mx::array& attention_mask,
    const mx::array& rotary_pos_emb)
{
    auto h = mx::add(hidden_states,
        attention_(mx::fast::rms_norm(hidden_states, norm1_weight_, eps_),
                   attention_mask, rotary_pos_emb));
    h = mx::add(h, mlp_(mx::fast::rms_norm(h, norm2_weight_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> Qwen25VLVisionBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["norm1.weight"] = &norm1_weight_;
    map["norm2.weight"] = &norm2_weight_;
    return map;
}

// -- Vision Model --

Qwen25VLVisionModel::Qwen25VLVisionModel(const Qwen25VLVisionConfiguration& config)
    : patch_embed_proj_weight_(mx::zeros({config.hidden_size, config.temporal_patch_size,
                                           config.patch_size, config.patch_size, config.in_channels})),
      patch_size_(config.patch_size),
      temporal_patch_size_(config.temporal_patch_size),
      in_channels_(config.in_channels),
      hidden_size_(config.hidden_size),
      rotary_pos_emb_(config.hidden_size / config.num_heads / 2, 10000.0f),
      merger_(config.out_hidden_size, config.hidden_size, config.spatial_merge_size),
      spatial_merge_size_(config.spatial_merge_size),
      spatial_merge_unit_(config.spatial_merge_size * config.spatial_merge_size),
      window_size_(config.window_size),
      fullatt_block_indexes_(config.fullatt_block_indexes)
{
    blocks_.reserve(config.depth);
    for (int i = 0; i < config.depth; ++i)
        blocks_.emplace_back(config);
}

mx::array Qwen25VLVisionModel::patch_embed(const mx::array& hidden_states) {
    // Reshape input: [N, C, T, H, W] -> [N, T, H, W, C] for Conv3d
    auto x = mx::reshape(hidden_states,
        {-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    // Move channels to last: [N, T, H, W, C]
    x = mx::transpose(x, {0, 2, 3, 4, 1});
    // Apply Conv3d as matmul with reshaped kernel
    int kernel_elements = temporal_patch_size_ * patch_size_ * patch_size_ * in_channels_;
    auto flat_x = mx::reshape(x, {-1, kernel_elements});
    auto flat_w = mx::reshape(patch_embed_proj_weight_, {hidden_size_, kernel_elements});
    auto out = mx::matmul(flat_x, mx::transpose(flat_w));
    return out; // [N, hidden_size]
}

mx::array Qwen25VLVisionModel::compute_rotary_pos_emb(const std::vector<THW>& frames) {
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

std::pair<mx::array, mx::array> Qwen25VLVisionModel::get_window_index(const std::vector<THW>& frames) {
    std::vector<mx::array> window_index_parts;
    std::vector<int> cu_window_seqlens = {0};
    int window_index_id = 0;
    int vit_merger_window_size = window_size_ / spatial_merge_size_ / patch_size_;

    for (const auto& frame : frames) {
        int grid_t = frame.t;
        int grid_h = frame.h;
        int grid_w = frame.w;
        int llm_grid_h = grid_h / spatial_merge_size_;
        int llm_grid_w = grid_w / spatial_merge_size_;

        // Create index array [0 .. grid_t * llm_grid_h * llm_grid_w)
        auto index = mx::reshape(
            mx::arange(0, grid_t * llm_grid_h * llm_grid_w, mx::int32),
            {grid_t, llm_grid_h, llm_grid_w});

        // Compute padding
        int pad_h = (vit_merger_window_size - (llm_grid_h % vit_merger_window_size)) % vit_merger_window_size;
        int pad_w = (vit_merger_window_size - (llm_grid_w % vit_merger_window_size)) % vit_merger_window_size;
        int num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
        int num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

        // Pad the index with -100
        mx::array index_padded = index;
        if (pad_h > 0 || pad_w > 0) {
            index_padded = mx::pad(index,
                {{0, 0}, {0, pad_h}, {0, pad_w}},
                mx::array(-100, mx::int32));
        } else {
            index_padded = index;
        }

        // Reshape and transpose for windowing
        auto index_reshaped = mx::reshape(index_padded,
            {grid_t, num_windows_h, vit_merger_window_size,
             num_windows_w, vit_merger_window_size});
        auto index_transposed = mx::transpose(index_reshaped, {0, 1, 3, 2, 4});
        index_transposed = mx::reshape(index_transposed,
            {grid_t, num_windows_h * num_windows_w,
             vit_merger_window_size, vit_merger_window_size});

        // Calculate sequence lengths per window (count of valid entries)
        auto valid_mask = mx::not_equal(index_transposed, mx::array(-100, mx::int32));
        auto seqlens = mx::sum(valid_mask, {2, 3});
        seqlens = mx::reshape(seqlens, {-1});

        // Flatten and extract valid indices
        auto index_flat = mx::flatten(index_transposed);

        // We need to evaluate to filter out -100 values
        mx::eval(index_flat);
        mx::eval(seqlens);

        // Extract valid (non -100) values
        int flat_size = index_flat.size();
        auto flat_data = index_flat.data<int32_t>();

        std::vector<int32_t> valid_values;
        valid_values.reserve(flat_size);
        for (int i = 0; i < flat_size; ++i) {
            if (flat_data[i] != -100) {
                valid_values.push_back(flat_data[i]);
            }
        }

        auto valid_arr = mx::array(valid_values.data(),
            {static_cast<int>(valid_values.size())}, mx::int32);

        // Add offset and append to window index
        window_index_parts.push_back(mx::add(valid_arr, mx::array(window_index_id, mx::int32)));

        // Update cumulative sequence lengths
        auto cu_seqlens_tmp = mx::multiply(
            mx::cumsum(seqlens, 0),
            mx::array(spatial_merge_unit_, mx::int32));
        cu_seqlens_tmp = mx::add(cu_seqlens_tmp,
            mx::array(cu_window_seqlens.back(), mx::int32));

        mx::eval(cu_seqlens_tmp);
        auto cu_data = cu_seqlens_tmp.data<int32_t>();
        int cu_size = cu_seqlens_tmp.size();
        for (int i = 0; i < cu_size; ++i) {
            cu_window_seqlens.push_back(cu_data[i]);
        }

        window_index_id += grid_t * llm_grid_h * llm_grid_w;
    }

    // Concatenate all window indices
    auto combined_window_index = mx::concatenate(window_index_parts, 0);

    // Deduplicate cu_window_seqlens while preserving order
    std::vector<int32_t> unique_cu;
    std::set<int> seen;
    for (int val : cu_window_seqlens) {
        if (seen.find(val) == seen.end()) {
            seen.insert(val);
            unique_cu.push_back(val);
        }
    }

    auto unique_cu_arr = mx::array(unique_cu.data(),
        {static_cast<int>(unique_cu.size())}, mx::int32);

    return {combined_window_index, unique_cu_arr};
}

mx::array Qwen25VLVisionModel::build_attention_mask(int sequence_length, const mx::array& cu_seqlens) {
    // Create attention mask filled with false
    auto mask = mx::zeros({1, sequence_length, sequence_length}, mx::bool_);

    // Evaluate cu_seqlens to read values
    mx::eval(cu_seqlens);
    auto cu_data = cu_seqlens.data<int32_t>();
    int num_seqlens = cu_seqlens.size();

    // Set true for each block-diagonal segment
    // Build the mask by creating individual block masks and combining them
    std::vector<mx::array> block_masks;

    for (int i = 1; i < num_seqlens; ++i) {
        int start = cu_data[i - 1];
        int end = cu_data[i];
        if (start >= end) continue;

        // Create a mask for this block
        auto row_idx = mx::arange(0, sequence_length, mx::int32);
        auto col_idx = mx::arange(0, sequence_length, mx::int32);

        // row >= start && row < end && col >= start && col < end
        auto row_ge = mx::greater_equal(row_idx, mx::array(start, mx::int32));
        auto row_lt = mx::less(row_idx, mx::array(end, mx::int32));
        auto row_in = mx::logical_and(row_ge, row_lt); // [S]

        auto col_ge = mx::greater_equal(col_idx, mx::array(start, mx::int32));
        auto col_lt = mx::less(col_idx, mx::array(end, mx::int32));
        auto col_in = mx::logical_and(col_ge, col_lt); // [S]

        // Outer product: [S, 1] * [1, S] -> [S, S]
        auto block = mx::logical_and(
            mx::expand_dims(row_in, 1),
            mx::expand_dims(col_in, 0));
        block_masks.push_back(block);
    }

    if (block_masks.empty()) {
        return mx::ones({1, sequence_length, sequence_length}, mx::bool_);
    }

    // Combine all block masks with logical OR
    auto combined = block_masks[0];
    for (size_t i = 1; i < block_masks.size(); ++i) {
        combined = mx::logical_or(combined, block_masks[i]);
    }

    return mx::expand_dims(combined, 0); // [1, S, S]
}

mx::array Qwen25VLVisionModel::operator()(const mx::array& hidden_states, const std::vector<THW>& frames) {
    auto h = patch_embed(hidden_states);
    auto rope = compute_rotary_pos_emb(frames);

    // Get window indices and cumulative sequence lengths
    auto [window_index, cu_window_seqlens] = get_window_index(frames);

    // Prepare attention masks
    int seq_len = h.shape(0);

    // Compute cumulative sequence lengths for full attention
    std::vector<int32_t> cu_seqlens_vec = {0};
    for (const auto& frame : frames) {
        int frame_seq_len = frame.h * frame.w;
        for (int ti = 0; ti < frame.t; ++ti) {
            cu_seqlens_vec.push_back(cu_seqlens_vec.back() + frame_seq_len);
        }
    }
    auto cu_seqlens = mx::array(cu_seqlens_vec.data(),
        {static_cast<int>(cu_seqlens_vec.size())}, mx::int32);

    auto full_attention_mask = build_attention_mask(seq_len, cu_seqlens);
    auto window_attention_mask = build_attention_mask(seq_len, cu_window_seqlens);

    // Reshape and reindex hidden states by window index
    h = mx::reshape(h, {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    h = mx::take(h, window_index, 0);
    h = mx::reshape(h, {seq_len, -1});

    // Reshape and reindex rotary position embeddings
    auto rope_reshaped = mx::reshape(rope, {seq_len / spatial_merge_unit_, spatial_merge_unit_, -1});
    rope_reshaped = mx::take(rope_reshaped, window_index, 0);
    rope_reshaped = mx::reshape(rope_reshaped, {seq_len, -1});

    // Check which blocks use full attention
    std::set<int> fullatt_set(fullatt_block_indexes_.begin(), fullatt_block_indexes_.end());

    // Process through blocks
    for (size_t i = 0; i < blocks_.size(); ++i) {
        // Use full attention for specific blocks, window attention for others
        const auto& attn_mask = fullatt_set.count(static_cast<int>(i)) > 0
            ? full_attention_mask : window_attention_mask;
        h = blocks_[i](h, attn_mask, rope_reshaped);
    }

    // Apply patch merger
    h = merger_(h);

    // Reorder back to original sequence
    auto reverse_indices = mx::argsort(window_index, 0);
    h = mx::take(h, reverse_indices, 0);

    return h;
}

std::unordered_map<std::string, mx::array> Qwen25VLVisionModel::sanitize(
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

std::unordered_map<std::string, mx::array*> Qwen25VLVisionModel::weight_map() {
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

Qwen25VLLanguageAttention::Qwen25VLLanguageAttention(const Qwen25VLTextConfiguration& args)
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

mx::array Qwen25VLLanguageAttention::operator()(
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

std::unordered_map<std::string, mx::array*> Qwen25VLLanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

Qwen25VLLanguageMLP::Qwen25VLLanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen25VLLanguageMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen25VLLanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Decoder Layer --

Qwen25VLDecoderLayer::Qwen25VLDecoderLayer(const Qwen25VLTextConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Qwen25VLDecoderLayer::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen25VLDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

Qwen25VLLanguageModelInner::Qwen25VLLanguageModelInner(const Qwen25VLTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Qwen25VLLanguageModelInner::operator()(
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

mx::array Qwen25VLLanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array Qwen25VLLanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen25VLLanguageModelInner::weight_map() {
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

Qwen25VLLanguageModel::Qwen25VLLanguageModel(const Qwen25VLTextConfiguration& args)
    : model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

LMOutput Qwen25VLLanguageModel::operator()(
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

std::unordered_map<std::string, mx::array*> Qwen25VLLanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Top-Level Model ────────────────────────────────────────────────────

Qwen25VLModel::Qwen25VLModel(const Qwen25VLConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array Qwen25VLModel::input_embeddings(
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

PrepareResult Qwen25VLModel::prepare_impl(
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

LMOutput Qwen25VLModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Qwen25VLModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array> Qwen25VLModel::sanitize_impl(
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

void Qwen25VLModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen25VLModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_tower_.weight_map()) map["vision_tower." + k] = v;
    for (auto& [k, v] : language_model_.weight_map()) map["language_model." + k] = v;
    return map;
}

} // namespace mlx_lm

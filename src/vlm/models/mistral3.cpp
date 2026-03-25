// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mistral3
// Mistral3 reuses Pixtral's vision model

#include <mlx-lm/vlm/models/mistral3.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Mistral3VLMTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("mistral3"));
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
    c.sliding_window = j.value("sliding_window", 0);
    c.use_qk_norm = j.value("use_qk_norm", false);

    if (j.contains("rope_parameters") && !j["rope_parameters"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> params;
        for (auto& [key, val] : j["rope_parameters"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            params[key] = sn;
        }
        c.rope_parameters = params;
    }

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }

    if (j.contains("layer_types") && j["layer_types"].is_array()) {
        c.layer_types = j["layer_types"].get<std::vector<std::string>>();
    }
    // If layer_types is empty, default to all full_attention
    if (c.layer_types.empty()) {
        c.layer_types.resize(c.num_hidden_layers, "full_attention");
    }
}

void from_json(const nlohmann::json& j, Mistral3VLMConfiguration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<Mistral3VLMTextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Mistral3VisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("mistral3"));
    c.ignore_index = j.value("ignore_index", -100);

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
    c.spatial_merge_size = j.value("spatial_merge_size", 2);
    c.multimodal_projector_bias = j.value("multimodal_projector_bias", false);
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

// Llama4 attention scaling: 1 + beta * log(1 + floor(position / max_position_embeddings))
static mx::array get_llama4_attention_scale(
    int start, int stop, float beta, int max_position_embeddings)
{
    if (beta == 0.0f) {
        // When beta is 0, scaling is just 1.0 for all positions
        int len = stop - start;
        std::vector<float> ones(len, 1.0f);
        auto result = mx::array(ones.data(), {len, 1}, mx::float32);
        return result;
    }

    // Build positions array
    int len = stop - start;
    std::vector<float> pos_data(len);
    for (int i = 0; i < len; ++i) {
        pos_data[i] = static_cast<float>(start + i);
    }
    auto positions = mx::array(pos_data.data(), {len}, mx::float32);

    // scaling = 1 + beta * log(1 + floor(positions / max_position_embeddings))
    auto floored = mx::floor(mx::divide(positions, mx::array(static_cast<float>(max_position_embeddings))));
    auto scaling = mx::add(
        mx::array(1.0f),
        mx::multiply(mx::array(beta), mx::log(mx::add(mx::array(1.0f), floored)))
    );
    return mx::expand_dims(scaling, -1);
}

// Unfold (im2col) operation: extract sliding local blocks
static mx::array unfold(
    const mx::array& input,
    int kernel_size,
    int dilation,
    int padding,
    int stride)
{
    auto x = input;
    int batch_size = x.shape(0);
    int channels = x.shape(1);
    int height = x.shape(2);
    int width = x.shape(3);

    // Add padding if needed
    if (padding > 0) {
        x = mx::pad(x, {{0, 0}, {0, 0}, {padding, padding}, {padding, padding}});
    }

    int padded_h = height + 2 * padding;
    int padded_w = width + 2 * padding;

    int height_out = (padded_h - dilation * (kernel_size - 1) - 1) / stride + 1;
    int width_out = (padded_w - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Extract blocks using nested loops
    // For each output position (i, j), collect kernel_size * kernel_size values
    std::vector<mx::array> blocks;
    blocks.reserve(height_out * width_out);

    for (int i = 0; i < padded_h - kernel_size * dilation + dilation; i += stride) {
        for (int j = 0; j < padded_w - kernel_size * dilation + dilation; j += stride) {
            // Collect kernel elements
            std::vector<mx::array> block_elements;
            block_elements.reserve(kernel_size * kernel_size);

            for (int di = 0; di < kernel_size; ++di) {
                for (int dj = 0; dj < kernel_size; ++dj) {
                    int h_idx = i + di * dilation;
                    int w_idx = j + dj * dilation;
                    // x[:, :, h_idx, w_idx] -> (B, C)
                    auto elem = mx::slice(x, {0, 0, h_idx, w_idx},
                                           {batch_size, channels, h_idx + 1, w_idx + 1});
                    elem = mx::reshape(elem, {batch_size, channels});
                    block_elements.push_back(elem);
                }
            }

            // Stack elements: (B, C, k*k) via transpose
            auto stacked = mx::stack(block_elements, 1);  // (B, k*k, C)
            stacked = mx::transpose(stacked, {0, 2, 1});  // (B, C, k*k)
            blocks.push_back(stacked);
        }
    }

    // Stack all blocks along last dim: (B, C, k*k, L)
    auto result = mx::stack(blocks, -1);

    // Reshape to (B, C*k*k, L)
    return mx::reshape(result, {batch_size, channels * kernel_size * kernel_size, height_out * width_out});
}

// ── Mistral3 Patch Merger ──────────────────────────────────────────────

Mistral3PatchMerger::Mistral3PatchMerger(int hidden_size, int spatial_merge_size, int patch_size)
    : spatial_merge_size_(spatial_merge_size),
      patch_size_(patch_size),
      hidden_size_(hidden_size),
      merging_layer_weight_(mx::zeros({hidden_size, hidden_size * spatial_merge_size * spatial_merge_size}))
{}

mx::array Mistral3PatchMerger::operator()(
    const mx::array& image_features,
    const std::vector<std::pair<int,int>>& image_sizes)
{
    // Convert image sizes to patch sizes
    std::vector<std::pair<int,int>> patch_sizes;
    patch_sizes.reserve(image_sizes.size());
    for (auto& [h, w] : image_sizes) {
        patch_sizes.push_back({h / patch_size_, w / patch_size_});
    }

    std::vector<int> tokens_per_image;
    tokens_per_image.reserve(patch_sizes.size());
    for (auto& [ph, pw] : patch_sizes) {
        tokens_per_image.push_back(ph * pw);
    }

    int d = image_features.shape(-1);
    auto features = mx::astype(image_features, mx::bfloat16);

    // Split features into per-image chunks
    // features is [1, total_tokens, d]
    std::vector<mx::array> chunks;
    if (tokens_per_image.size() == 1) {
        // Single image: take all tokens from batch dim 0
        chunks.push_back(mx::slice(features, {0, 0, 0},
                                    {1, features.shape(1), features.shape(2)}));
        chunks.back() = mx::reshape(chunks.back(), {features.shape(1), d});
    } else {
        // Multiple images: split along token dimension
        auto features_flat = mx::reshape(features, {features.shape(1), d});
        int current = 0;
        for (size_t i = 0; i < tokens_per_image.size(); ++i) {
            int count = tokens_per_image[i];
            chunks.push_back(mx::slice(features_flat, {current, 0}, {current + count, d}));
            current += count;
        }
    }

    std::vector<mx::array> permuted_tensors;

    for (size_t img_idx = 0; img_idx < chunks.size(); ++img_idx) {
        auto& image_tokens = chunks[img_idx];
        if (image_tokens.shape(0) > 0) {
            auto [ph, pw] = patch_sizes[img_idx];

            // Reshape to grid: (ph*pw, d) -> (ph, pw, d) -> (d, ph, pw) -> (1, d, ph, pw)
            auto image_grid = mx::reshape(image_tokens, {ph, pw, d});
            image_grid = mx::transpose(image_grid, {2, 0, 1});
            image_grid = mx::expand_dims(image_grid, 0);

            // Apply unfold
            auto grid = unfold(image_grid, spatial_merge_size_, 1, 0, spatial_merge_size_);

            // Reshape: (d * sms^2, -1).T
            grid = mx::reshape(grid, {d * spatial_merge_size_ * spatial_merge_size_, -1});
            grid = mx::transpose(grid);
            permuted_tensors.push_back(grid);
        }
    }

    auto merged = mx::concatenate(permuted_tensors, 0);

    // Apply merging layer linear (no bias)
    merged = linear_fwd(merged, merging_layer_weight_);

    // Add batch dimension back: (1, num_merged_tokens, hidden_size)
    return mx::expand_dims(merged, 0);
}

std::unordered_map<std::string, mx::array*> Mistral3PatchMerger::weight_map() {
    return {
        {"merging_layer.weight", &merging_layer_weight_},
    };
}

// ── Mistral3 MultiModal Projector ──────────────────────────────────────

Mistral3MultiModalProjector::Mistral3MultiModalProjector(const Mistral3VLMConfiguration& config)
    : norm_weight_(mx::ones({config.vision_config.hidden_size})),
      rms_norm_eps_(config.vision_config.rms_norm_eps),
      patch_merger_(config.vision_config.hidden_size, config.spatial_merge_size,
                    config.vision_config.patch_size),
      linear_1_weight_(mx::zeros({config.text_config.hidden_size, config.vision_config.hidden_size})),
      linear_2_weight_(mx::zeros({config.text_config.hidden_size, config.text_config.hidden_size}))
{
    if (config.multimodal_projector_bias) {
        linear_1_bias_ = mx::zeros({config.text_config.hidden_size});
        linear_2_bias_ = mx::zeros({config.text_config.hidden_size});
    }
}

mx::array Mistral3MultiModalProjector::operator()(
    const mx::array& x,
    const std::vector<std::pair<int,int>>& image_sizes)
{
    auto result = mx::fast::rms_norm(x, norm_weight_, rms_norm_eps_);
    result = patch_merger_(result, image_sizes);
    const mx::array* b1 = linear_1_bias_.has_value() ? &linear_1_bias_.value() : nullptr;
    result = linear_fwd(result, linear_1_weight_, b1);
    result = gelu(result);
    const mx::array* b2 = linear_2_bias_.has_value() ? &linear_2_bias_.value() : nullptr;
    result = linear_fwd(result, linear_2_weight_, b2);
    return result;
}

std::unordered_map<std::string, mx::array*> Mistral3MultiModalProjector::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["norm.weight"] = &norm_weight_;
    for (auto& [k, v] : patch_merger_.weight_map()) map["patch_merger." + k] = v;
    map["linear_1.weight"] = &linear_1_weight_;
    map["linear_2.weight"] = &linear_2_weight_;
    if (linear_1_bias_.has_value()) map["linear_1.bias"] = &linear_1_bias_.value();
    if (linear_2_bias_.has_value()) map["linear_2.bias"] = &linear_2_bias_.value();
    return map;
}

// ── Language Components ────────────────────────────────────────────────

// -- Language Attention --

Mistral3LanguageAttention::Mistral3LanguageAttention(const Mistral3VLMTextConfiguration& args)
    : heads_(args.num_attention_heads),
      kv_heads_(args.effective_num_kv_heads()),
      head_dim_(args.effective_head_dim()),
      scale_(std::pow(static_cast<float>(args.effective_head_dim()), -0.5f)),
      rope_traditional_(false),
      rope_scale_(1.0f),
      wq_weight_(mx::zeros({args.num_attention_heads * args.effective_head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.effective_num_kv_heads() * args.effective_head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.effective_num_kv_heads() * args.effective_head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.effective_head_dim()}))
{
    // Extract rope_theta from rope_parameters (required)
    if (args.rope_parameters.has_value()) {
        auto& params = args.rope_parameters.value();
        auto it = params.find("rope_theta");
        if (it != params.end() && it->second.is_float()) {
            rope_theta_ = it->second.as_float();
        } else {
            rope_theta_ = args.rope_theta;
        }
    } else {
        rope_theta_ = args.rope_theta;
    }
}

mx::array Mistral3LanguageAttention::operator()(
    const mx::array& x,
    const mx::array& attention_scale,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);
    keys    = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);

    // Apply Llama4 attention scaling to queries
    // attention_scale shape: (L, 1), need to broadcast to (1, 1, L, 1)
    auto scale_expanded = mx::expand_dims(mx::expand_dims(attention_scale, 0), 0);
    queries = mx::multiply(queries, mx::astype(scale_expanded, queries.dtype()));

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Mistral3LanguageAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// -- Language MLP --

Mistral3LanguageMLP::Mistral3LanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Mistral3LanguageMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_),
                             linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Mistral3LanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Transformer Block --

Mistral3TransformerBlock::Mistral3TransformerBlock(
    const Mistral3VLMTextConfiguration& args, bool use_sliding)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      use_sliding_(use_sliding)
{}

mx::array Mistral3TransformerBlock::operator()(
    const mx::array& x,
    const mx::array& attention_scale,
    const AttentionMask& mask,
    KVCache* cache)
{
    auto r = attention_(
        mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_),
        attention_scale, mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Mistral3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// -- Language Model Inner --

Mistral3LanguageModelInner::Mistral3LanguageModelInner(const Mistral3VLMTextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      layer_types_(args.layer_types),
      sliding_window_(args.sliding_window)
{
    // Build layers based on layer_types
    layers_.reserve(layer_types_.size());
    for (auto& lt : layer_types_) {
        layers_.emplace_back(args, lt == "sliding_attention");
    }

    // Find first full_attention and first sliding_attention indices
    fa_index_ = 0;
    swa_index_ = -1;
    for (size_t i = 0; i < layer_types_.size(); ++i) {
        if (layer_types_[i] == "full_attention") {
            fa_index_ = static_cast<int>(i);
            break;
        }
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].is_sliding()) {
            swa_index_ = static_cast<int>(i);
            break;
        }
    }

    // Extract Llama4 scaling parameters from rope_parameters
    llama4_beta_ = 0.0f;
    original_max_pos_ = args.max_position_embeddings > 0 ? args.max_position_embeddings : 4096;

    if (args.rope_parameters.has_value()) {
        auto& params = args.rope_parameters.value();
        auto beta_it = params.find("llama_4_scaling_beta");
        if (beta_it != params.end() && beta_it->second.is_float()) {
            llama4_beta_ = beta_it->second.as_float();
        }
        auto omp_it = params.find("original_max_position_embeddings");
        if (omp_it != params.end() && omp_it->second.is_float()) {
            original_max_pos_ = static_cast<int>(omp_it->second.as_float());
        }
    }
}

mx::array Mistral3LanguageModelInner::operator()(
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

    int offset = (cache && !cache->empty()) ? (*cache)[0].offset() : 0;

    // Create full attention mask
    auto fa_mask = create_attention_mask(
        h, (cache && fa_index_ < static_cast<int>(cache->size())) ? &(*cache)[fa_index_] : nullptr);

    // Create sliding window attention mask
    AttentionMask swa_mask;
    if (swa_index_ >= 0 && sliding_window_ > 0 && cache && !cache->empty()) {
        int t = h.shape(1);
        if (t > 1) {
            int swa_offset = std::min(sliding_window_, (*cache)[swa_index_].offset());
            swa_mask = AttentionMask::from_array(create_causal_mask(t, swa_offset, sliding_window_));
        }
    }

    // Compute Llama4 attention scale
    auto attention_scale = get_llama4_attention_scale(
        offset, offset + h.shape(1), llama4_beta_, original_max_pos_);
    attention_scale = mx::astype(attention_scale, h.dtype());

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto& mask = layers_[i].is_sliding() ? swa_mask : fa_mask;
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, attention_scale, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Mistral3LanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array Mistral3LanguageModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Mistral3LanguageModelInner::weight_map() {
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

Mistral3LanguageModel::Mistral3LanguageModel(const Mistral3VLMTextConfiguration& args)
    : model_(args),
      tie_word_embeddings_(args.tie_word_embeddings),
      layer_types_(args.layer_types),
      sliding_window_(args.sliding_window)
{
    kv_heads_.resize(args.num_hidden_layers, args.effective_num_kv_heads());
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

LMOutput Mistral3LanguageModel::operator()(
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

std::vector<KVCache> Mistral3LanguageModel::new_cache(const GenerateParameters& params) const {
    std::vector<KVCache> caches;
    caches.reserve(layer_types_.size());

    for (auto& lt : layer_types_) {
        if (lt == "sliding_attention" && sliding_window_ > 0) {
            caches.emplace_back(RotatingKVCache(sliding_window_));
        } else if (params.max_kv_size.has_value()) {
            caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
        } else {
            caches.emplace_back(KVCacheSimple{});
        }
    }

    return caches;
}

std::unordered_map<std::string, mx::array*> Mistral3LanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

// ── Top-Level Mistral3 Model ──────────────────────────────────────────

Mistral3Model::Mistral3Model(const Mistral3VLMConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config),
      multi_modal_projector_(config),
      vision_feature_layer_(config.vision_feature_layer)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array Mistral3Model::get_input_embeddings(
    const mx::array& input_ids,
    const mx::array* pixel_values,
    const std::vector<std::pair<int,int>>* image_sizes)
{
    if (!pixel_values || !image_sizes) {
        return language_model_.inner().embed_tokens(input_ids);
    }

    auto inputs_embeds = language_model_.inner().embed_tokens(input_ids);

    // Handle 3D pixel values (missing batch dimension)
    auto pv = *pixel_values;
    if (pv.ndim() == 3) {
        pv = mx::expand_dims(pv, 0);
    }

    // Process through vision tower with hidden states (reuses Pixtral vision model)
    auto vision_out = vision_tower_(pv, /*output_hidden_states=*/true);

    // Select features from specified layer
    auto& hidden_states = vision_out.hidden_states;
    int layer_index = vision_feature_layer_ < 0
        ? static_cast<int>(hidden_states.size()) + vision_feature_layer_
        : vision_feature_layer_;

    auto selected_features = hidden_states[layer_index];

    // Project to text space using Mistral3's patch merger projector
    auto image_features = multi_modal_projector_(selected_features, *image_sizes);

    // Merge embeddings: find image token positions and interleave
    int num_image_patches = image_features.shape(1);

    auto input_ids_flat = input_ids.ndim() > 1
        ? mx::reshape(input_ids, {-1})
        : input_ids;

    int seq_len = input_ids_flat.shape(0);
    mx::eval(input_ids_flat);
    const int32_t* ids_data = input_ids_flat.data<int32_t>();

    std::vector<int> image_positions;
    for (int i = 0; i < seq_len; ++i) {
        if (ids_data[i] == static_cast<int32_t>(config_.image_token_index)) {
            image_positions.push_back(i);
        }
    }

    auto embeds = inputs_embeds;
    if (embeds.ndim() == 2) {
        embeds = mx::expand_dims(embeds, 0);
    }

    // Build text segments and interleave with image embeddings
    std::vector<mx::array> final_embeddings;
    int start_idx = 0;

    // Split image features into individual patches along axis 1
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

PrepareResult Mistral3Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    const mx::array* pixel_values = nullptr;
    mx::array pixels = mx::array(0.0f);
    const std::vector<std::pair<int,int>>* image_sizes = nullptr;
    std::vector<std::pair<int,int>> sizes;

    if (input.image.has_value()) {
        pixels = input.image->pixels;
        pixel_values = &pixels;

        // Extract image sizes from frames
        if (input.image->frames.has_value()) {
            for (auto& f : input.image->frames.value()) {
                sizes.push_back({f.h, f.w});
            }
            image_sizes = &sizes;
        } else {
            sizes.push_back({config_.vision_config.image_size, config_.vision_config.image_size});
            image_sizes = &sizes;
        }
    }

    auto embeddings = get_input_embeddings(input_ids, pixel_values, image_sizes);

    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, embeddings);

    return PrepareResult::logits(std::move(result));
}

LMOutput Mistral3Model::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Mistral3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::vector<KVCache> Mistral3Model::new_cache_impl(const GenerateParameters& params) const {
    return language_model_.new_cache(params);
}

std::unordered_map<std::string, mx::array>
Mistral3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
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

        // Handle weight scale patterns
        if (new_key.find("weight_scale_inv") != std::string::npos) {
            auto scale_inv = value;
            auto weight_key = new_key;
            auto pos = weight_key.find("_scale_inv");
            if (pos != std::string::npos) {
                weight_key.erase(pos, 10);  // Remove "_scale_inv"
            }
            // Find the corresponding weight key in the original weights
            auto orig_weight_key = key;
            auto orig_pos = orig_weight_key.find("_scale_inv");
            if (orig_pos != std::string::npos) {
                orig_weight_key.erase(orig_pos, 10);
            }
            auto wit = weights.find(orig_weight_key);
            if (wit != weights.end()) {
                new_weights.insert_or_assign(weight_key, mx::multiply(wit->second, scale_inv));
            }
        } else if (new_key.find("activation_scale") != std::string::npos) {
            continue;
        } else if (new_weights.find(new_key) == new_weights.end()) {
            new_weights.insert_or_assign(new_key, value);
        }
    }

    // Sanitize vision conv weights
    return vision_tower_.sanitize(std::move(new_weights));
}

void Mistral3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Mistral3Model::weight_map() {
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

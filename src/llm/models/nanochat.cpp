// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of NanoChat.swift

#include <mlx-lm/llm/models/nanochat.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, NanoChatConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.vocab_size = j.at("vocab_size").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
    c.logits_softcap = j.value("logits_softcap", 15.0f);
}

// --- Helpers ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// Functional RMS norm — no learned weight parameter
static mx::array functional_rms_norm(const mx::array& x, float eps) {
    auto mean_sq = mx::mean(mx::square(x), /* axis= */ -1, /* keepdims= */ true);
    return mx::multiply(x, mx::rsqrt(mx::add(mean_sq, mx::array(eps))));
}

// Softcap now uses compiled logit_softcap() from activations.h

// --- NanoChatAttention ---

NanoChatAttention::NanoChatAttention(const NanoChatConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      rms_norm_eps_(config.rms_norm_eps),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      rope_freqs_(mx::array(0.0f)) // placeholder — computed below
{
    // Precompute RoPE frequencies:
    // freqScale = log(rope_theta) / halfDim
    // freqs = -exp(indices * freqScale)
    int half_dim = head_dim_ / 2;
    float freq_scale = std::log(config.rope_theta) / static_cast<float>(half_dim);
    std::vector<float> freq_data(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        freq_data[i] = -std::exp(static_cast<float>(i) * freq_scale);
    }
    rope_freqs_ = mx::array(freq_data.data(), {half_dim});
}

mx::array NanoChatAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // Apply RoPE with precomputed frequencies
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, std::nullopt, 1.0f, offset, rope_freqs_);
    keys = mx::fast::rope(keys, head_dim_, false, std::nullopt, 1.0f, offset, rope_freqs_);

    // QK norm — functional rms_norm applied to queries and keys after RoPE
    queries = functional_rms_norm(queries, rms_norm_eps_);
    keys = functional_rms_norm(keys, rms_norm_eps_);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // Scaled dot-product attention
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> NanoChatAttention::weight_map() {
    return {
        {"c_q.weight", &wq_weight_},
        {"c_k.weight", &wk_weight_},
        {"c_v.weight", &wv_weight_},
        {"c_proj.weight", &wo_weight_},
    };
}

// --- NanoChatMLP ---

NanoChatMLP::NanoChatMLP(const NanoChatConfiguration& config)
    : fc_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      proj_weight_(mx::zeros({config.hidden_size, config.intermediate_size}))
{}

mx::array NanoChatMLP::operator()(const mx::array& x) {
    // Squared ReLU: proj(relu(fc(x)) * relu(fc(x)))
    auto activated = mx::maximum(linear_fwd(x, fc_weight_), mx::array(0.0f));
    return linear_fwd(mx::multiply(activated, activated), proj_weight_);
}

std::unordered_map<std::string, mx::array*> NanoChatMLP::weight_map() {
    return {
        {"c_fc.weight", &fc_weight_},
        {"c_proj.weight", &proj_weight_},
    };
}

// --- NanoChatBlock ---

NanoChatBlock::NanoChatBlock(const NanoChatConfiguration& config)
    : attn_(config),
      mlp_(config),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array NanoChatBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Pre-attention functional RMSNorm, then residual
    auto attn_out = attn_(functional_rms_norm(x, rms_norm_eps_), mask, cache);
    auto residual = mx::add(x, attn_out);

    // Pre-MLP functional RMSNorm, then residual
    auto mlp_out = mlp_(functional_rms_norm(residual, rms_norm_eps_));
    return mx::add(residual, mlp_out);
}

std::unordered_map<std::string, mx::array*> NanoChatBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attn_.weight_map()) map["attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    return map;
}

// --- NanoChatModelInner ---

NanoChatModelInner::NanoChatModelInner(const NanoChatConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array NanoChatModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    // Embedding lookup
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Apply functional RMSNorm after embedding
    h = functional_rms_norm(h, rms_norm_eps_);

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    // Final functional RMSNorm
    return functional_rms_norm(h, rms_norm_eps_);
}

std::unordered_map<std::string, mx::array*> NanoChatModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["wte.weight"] = &embed_tokens_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "h." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }
    return map;
}

// --- NanoChatModel ---

NanoChatModel::NanoChatModel(const NanoChatConfiguration& config)
    : config_(config),
      transformer_(config),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult NanoChatModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput NanoChatModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array NanoChatModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = transformer_(inputs, cache);
    auto logits = linear_fwd(out, lm_head_weight_);
    return logit_softcap(logits, config_.logits_softcap);
}

std::unordered_map<std::string, mx::array>
NanoChatModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    return weights;
}

void NanoChatModel::load_weights(
    const std::unordered_map<std::string, mx::array>& weights)
{
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
        }
    }
}

std::unordered_map<std::string, mx::array*> NanoChatModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : transformer_.weight_map()) {
        map["transformer." + k] = v;
    }
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

} // namespace mlx_lm

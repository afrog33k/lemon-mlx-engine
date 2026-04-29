// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/qwen3.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <cstdlib>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, Qwen3Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.head_dim = j.at("head_dim").get<int>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

static bool project_only_last_token_for_generation() {
    const char* raw = std::getenv("LEMON_MLX_FULL_PREFILL_LOGITS");
    return !(raw && raw[0] == '1' && raw[1] == '\0');
}

static mx::array last_token_hidden_for_generation(const mx::array& x) {
    if (!project_only_last_token_for_generation() || x.ndim() != 3 || x.shape(1) <= 1) {
        return x;
    }
    int seq_len = x.shape(1);
    return mx::slice(x, {0, seq_len - 1, 0}, {x.shape(0), seq_len, x.shape(2)});
}

// --- Qwen3Attention ---

Qwen3Attention::Qwen3Attention(const Qwen3Configuration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      scale_(std::pow(static_cast<float>(args.head_dim), -0.5f)),
      wqkv_weight_(mx::zeros({
          (args.num_attention_heads + 2 * args.num_key_value_heads) * args.head_dim,
          args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim})),
      q_norm_weight_(mx::ones({args.head_dim})),
      k_norm_weight_(mx::ones({args.head_dim})),
      rms_norm_eps_(args.rms_norm_eps),
      rope_theta_(args.rope_theta),
      rope_scale_(1.0f)
{
    if (args.rope_scaling.has_value()) {
        auto& scaling = args.rope_scaling.value();
        auto type_it = scaling.find("type");
        if (type_it != scaling.end() && type_it->second.is_string() && type_it->second.as_string() == "linear") {
            auto factor_it = scaling.find("factor");
            if (factor_it != scaling.end() && factor_it->second.is_float()) {
                rope_scale_ = 1.0f / factor_it->second.as_float();
            }
        }
    }
}

mx::array Qwen3Attention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto qkv = linear_fwd(x, wqkv_weight_);
    int q_dim = num_heads_ * head_dim_;
    int kv_dim = num_kv_heads_ * head_dim_;
    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, q_dim});
    auto keys = mx::slice(qkv, {0, 0, q_dim}, {B, L, q_dim + kv_dim});
    auto values = mx::slice(qkv, {0, 0, q_dim + kv_dim}, {B, L, q_dim + 2 * kv_dim});

    // Reshape and apply Q/K norms before transpose
    queries = mx::reshape(queries, {B, L, num_heads_, -1});
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3});

    keys = mx::reshape(keys, {B, L, num_kv_heads_, -1});
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3Attention::weight_map() {
    return {
        {"qkv_proj.weight", &wqkv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- Qwen3MLP ---

Qwen3MLP::Qwen3MLP(int dimensions, int hidden_dimensions)
    : gate_up_weight_(mx::zeros({2 * hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions}))
{}

mx::array Qwen3MLP::operator()(const mx::array& x) {
    auto gate_up = linear_fwd(x, gate_up_weight_);
    auto parts = mx::split(gate_up, 2, -1);
    return linear_fwd(swiglu(parts[0], parts[1]), down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3MLP::weight_map() {
    return {
        {"gate_up_proj.weight", &gate_up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- Qwen3TransformerBlock ---

Qwen3TransformerBlock::Qwen3TransformerBlock(const Qwen3Configuration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Qwen3TransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Qwen3ModelInner ---

Qwen3ModelInner::Qwen3ModelInner(const Qwen3Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Qwen3ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = embedding_forward(embed_tokens_weight_, inputs);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen3ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_fwd(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Qwen3Model ---

Qwen3Model::Qwen3Model(const Qwen3Configuration& args)
    : config_(args), model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Qwen3Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Qwen3Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Qwen3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    out = last_token_hidden_for_generation(out);
    if (lm_head_weight_.has_value()) return linear_fwd(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Qwen3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string attn_prefix = "model.layers." + std::to_string(l) + ".self_attn.";
        std::string q_prefix = attn_prefix + "q_proj";
        std::string k_prefix = attn_prefix + "k_proj";
        std::string v_prefix = attn_prefix + "v_proj";
        std::string qkv_prefix = attn_prefix + "qkv_proj";

        auto q_weight = weights.find(q_prefix + ".weight");
        auto k_weight = weights.find(k_prefix + ".weight");
        auto v_weight = weights.find(v_prefix + ".weight");
        if (q_weight != weights.end() && k_weight != weights.end() && v_weight != weights.end()) {
            std::vector<mx::array> joined;
            joined.push_back(std::move(q_weight->second));
            joined.push_back(std::move(k_weight->second));
            joined.push_back(std::move(v_weight->second));
            weights.erase(q_weight);
            weights.erase(k_weight);
            weights.erase(v_weight);
            weights.insert_or_assign(qkv_prefix + ".weight", mx::concatenate(joined, 0));

            auto q_scales = weights.find(q_prefix + ".scales");
            auto k_scales = weights.find(k_prefix + ".scales");
            auto v_scales = weights.find(v_prefix + ".scales");
            if (q_scales != weights.end() && k_scales != weights.end() && v_scales != weights.end()) {
                joined.clear();
                joined.push_back(std::move(q_scales->second));
                joined.push_back(std::move(k_scales->second));
                joined.push_back(std::move(v_scales->second));
                weights.erase(q_scales);
                weights.erase(k_scales);
                weights.erase(v_scales);
                weights.insert_or_assign(qkv_prefix + ".scales", mx::concatenate(joined, 0));
            }

            auto q_biases = weights.find(q_prefix + ".biases");
            auto k_biases = weights.find(k_prefix + ".biases");
            auto v_biases = weights.find(v_prefix + ".biases");
            if (q_biases != weights.end() && k_biases != weights.end() && v_biases != weights.end()) {
                joined.clear();
                joined.push_back(std::move(q_biases->second));
                joined.push_back(std::move(k_biases->second));
                joined.push_back(std::move(v_biases->second));
                weights.erase(q_biases);
                weights.erase(k_biases);
                weights.erase(v_biases);
                weights.insert_or_assign(qkv_prefix + ".biases", mx::concatenate(joined, 0));
            }
        }

        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        std::string gate_prefix = prefix + "gate_proj";
        std::string up_prefix = prefix + "up_proj";
        std::string gate_up_prefix = prefix + "gate_up_proj";

        auto gate_weight = weights.find(gate_prefix + ".weight");
        auto up_weight = weights.find(up_prefix + ".weight");
        if (gate_weight == weights.end() || up_weight == weights.end()) {
            continue;
        }

        std::vector<mx::array> joined;
        joined.push_back(std::move(gate_weight->second));
        joined.push_back(std::move(up_weight->second));
        weights.erase(gate_weight);
        weights.erase(up_weight);
        weights.insert_or_assign(gate_up_prefix + ".weight", mx::concatenate(joined, 0));

        auto gate_scales = weights.find(gate_prefix + ".scales");
        auto up_scales = weights.find(up_prefix + ".scales");
        if (gate_scales != weights.end() && up_scales != weights.end()) {
            joined.clear();
            joined.push_back(std::move(gate_scales->second));
            joined.push_back(std::move(up_scales->second));
            weights.erase(gate_scales);
            weights.erase(up_scales);
            weights.insert_or_assign(gate_up_prefix + ".scales", mx::concatenate(joined, 0));
        }

        auto gate_biases = weights.find(gate_prefix + ".biases");
        auto up_biases = weights.find(up_prefix + ".biases");
        if (gate_biases != weights.end() && up_biases != weights.end()) {
            joined.clear();
            joined.push_back(std::move(gate_biases->second));
            joined.push_back(std::move(up_biases->second));
            weights.erase(gate_biases);
            weights.erase(up_biases);
            weights.insert_or_assign(gate_up_prefix + ".biases", mx::concatenate(joined, 0));
        }
    }

    return weights;
}

void Qwen3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen3Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm

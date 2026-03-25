// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/embedders/nomic_bert.h>
#include <mlx-lm/common/activations.h>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, NomicBertConfiguration& c) {
    c.layer_norm_eps = j.value("layer_norm_epsilon", 1e-12f);
    c.mlp_fc1_bias = j.value("mlp_fc1_bias", false);
    c.mlp_fc2_bias = j.value("mlp_fc2_bias", false);
    c.embed_dim = j.value("n_embd", 768);
    c.num_heads = j.value("n_head", 12);
    c.mlp_dim = j.value("n_inner", 3072);
    c.num_layers = j.value("n_layer", 12);
    c.qkv_proj_bias = j.value("qkv_proj_bias", false);
    c.rotary_emb_base = j.value("rotary_emb_base", 1000.0f);
    c.rotary_emb_fraction = j.value("rotary_emb_fraction", 1.0f);
    c.rotary_emb_interleaved = j.value("rotary_emb_interleaved", false);
    if (j.contains("rotary_scaling_factor") && !j["rotary_scaling_factor"].is_null()) {
        c.rotary_scaling_factor = j["rotary_scaling_factor"].get<float>();
    }
    c.type_vocabulary_size = j.value("type_vocab_size", 2);
    c.vocabulary_size = j.value("vocab_size", 30528);
    c.max_position_embeddings = j.value("max_position_embeddings", 0);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

static mx::array layer_norm(const mx::array& x, const mx::array& w,
                              const mx::array& b, float eps) {
    return mx::fast::layer_norm(x, w, b, eps);
}

// --- NomicEmbedding ---

NomicEmbedding::NomicEmbedding(const NomicBertConfiguration& config)
    : word_embeddings_weight_(mx::zeros({config.vocabulary_size, config.embed_dim})),
      norm_weight_(mx::ones({config.embed_dim})),
      norm_bias_(mx::zeros({config.embed_dim})),
      norm_eps_(config.layer_norm_eps),
      type_vocabulary_size_(config.type_vocabulary_size)
{
    if (config.type_vocabulary_size > 0) {
        token_type_embeddings_weight_ = mx::zeros({config.type_vocabulary_size, config.embed_dim});
    }
    if (config.max_position_embeddings > 0) {
        position_embeddings_weight_ = mx::zeros({config.max_position_embeddings, config.embed_dim});
    }
}

mx::array NomicEmbedding::operator()(const mx::array& input_ids,
                                      const std::optional<mx::array>& position_ids,
                                      const std::optional<mx::array>& token_type_ids) {
    auto words = mx::take(word_embeddings_weight_, input_ids, 0);

    if (token_type_ids.has_value() && token_type_embeddings_weight_.has_value()) {
        words = mx::add(words, mx::take(token_type_embeddings_weight_.value(), token_type_ids.value(), 0));
    }

    int seq_len = input_ids.shape(1);
    auto pos_ids = position_ids.has_value() ? position_ids.value()
        : mx::broadcast_to(mx::arange(seq_len), input_ids.shape());

    if (position_embeddings_weight_.has_value()) {
        words = mx::add(words, mx::take(position_embeddings_weight_.value(), pos_ids, 0));
    }

    return layer_norm(words, norm_weight_, norm_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> NomicEmbedding::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["word_embeddings.weight"] = &word_embeddings_weight_;
    map["norm.weight"] = &norm_weight_;
    map["norm.bias"] = &norm_bias_;
    if (token_type_embeddings_weight_.has_value())
        map["token_type_embeddings.weight"] = &token_type_embeddings_weight_.value();
    if (position_embeddings_weight_.has_value())
        map["position_embeddings.weight"] = &position_embeddings_weight_.value();
    return map;
}

// --- NomicMLP ---

NomicMLP::NomicMLP(const NomicBertConfiguration& config)
    : up_weight_(mx::zeros({((config.mlp_dim + 255) / 256 * 256), config.embed_dim})),
      gate_weight_(mx::zeros({((config.mlp_dim + 255) / 256 * 256), config.embed_dim})),
      down_weight_(mx::zeros({config.embed_dim, ((config.mlp_dim + 255) / 256 * 256)}))
{
    int multiple_of = 256;
    int hidden = (config.mlp_dim + multiple_of - 1) / multiple_of * multiple_of;
    (void)hidden; // used in optional bias init below
    if (config.mlp_fc1_bias) {
        up_bias_ = mx::zeros({hidden});
        gate_bias_ = mx::zeros({hidden});
    }
    if (config.mlp_fc2_bias) {
        down_bias_ = mx::zeros({config.embed_dim});
    }
}

mx::array NomicMLP::operator()(const mx::array& x) {
    auto up = linear_fwd(x, up_weight_, up_bias_.has_value() ? &up_bias_.value() : nullptr);
    auto gate = linear_fwd(x, gate_weight_, gate_bias_.has_value() ? &gate_bias_.value() : nullptr);
    auto activation = swiglu(gate, up);
    return linear_fwd(activation, down_weight_, down_bias_.has_value() ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NomicMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["fc11.weight"] = &up_weight_;
    map["fc12.weight"] = &gate_weight_;
    map["fc2.weight"] = &down_weight_;
    if (up_bias_.has_value()) map["fc11.bias"] = &up_bias_.value();
    if (gate_bias_.has_value()) map["fc12.bias"] = &gate_bias_.value();
    if (down_bias_.has_value()) map["fc2.bias"] = &down_bias_.value();
    return map;
}

// --- NomicAttention ---

NomicAttention::NomicAttention(const NomicBertConfiguration& config)
    : num_heads_(config.num_heads),
      head_dim_(config.embed_dim / config.num_heads),
      rotary_emb_dim_(static_cast<int>(static_cast<float>(config.embed_dim / config.num_heads) * config.rotary_emb_fraction)),
      norm_factor_(std::sqrt(static_cast<float>(config.embed_dim / config.num_heads))),
      wqkv_weight_(mx::zeros({3 * config.embed_dim, config.embed_dim})),
      wo_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      use_dynamic_ntk_(config.rotary_scaling_factor.has_value()),
      dims_(rotary_emb_dim_),
      max_position_embeddings_(config.max_position_embeddings > 0
          ? std::optional<int>(config.max_position_embeddings) : std::nullopt),
      traditional_(config.rotary_emb_interleaved),
      base_(config.rotary_emb_base),
      scale_(config.rotary_scaling_factor.value_or(1.0f))
{
    if (config.qkv_proj_bias) {
        wqkv_bias_ = mx::zeros({3 * config.embed_dim});
        wo_bias_ = mx::zeros({config.embed_dim});
    }
}

mx::array NomicAttention::operator()(const mx::array& inputs,
                                      const std::optional<mx::array>& mask) {
    int B = inputs.shape(0), L = inputs.shape(1);
    int qp = num_heads_ * head_dim_;

    auto qkv = linear_fwd(inputs, wqkv_weight_, wqkv_bias_.has_value() ? &wqkv_bias_.value() : nullptr);

    // Split into Q, K, V
    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, qp});
    auto keys = mx::slice(qkv, {0, 0, qp}, {B, L, 2 * qp});
    auto values = mx::slice(qkv, {0, 0, 2 * qp}, {B, L, 3 * qp});

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_heads_, -1}), {0, 2, 1, 3});

    if (rotary_emb_dim_ > 0) {
        queries = mx::fast::rope(queries, dims_, traditional_, base_, scale_, 0);
        keys = mx::fast::rope(keys, dims_, traditional_, base_, scale_, 0);
    }

    auto scores = mx::divide(mx::matmul(queries, mx::transpose(keys, {0, 1, 3, 2})),
                               mx::array(norm_factor_));
    if (mask.has_value()) {
        scores = mx::add(scores, mask.value());
    }
    auto probs = mx::softmax(scores, -1);

    auto output = mx::matmul(probs, values);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, wo_bias_.has_value() ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NomicAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["Wqkv.weight"] = &wqkv_weight_;
    map["out_proj.weight"] = &wo_weight_;
    if (wqkv_bias_.has_value()) map["Wqkv.bias"] = &wqkv_bias_.value();
    if (wo_bias_.has_value()) map["out_proj.bias"] = &wo_bias_.value();
    return map;
}

// --- NomicTransformerBlock ---

NomicTransformerBlock::NomicTransformerBlock(const NomicBertConfiguration& config)
    : attention_(config),
      mlp_(config),
      norm1_weight_(mx::ones({config.embed_dim})),
      norm1_bias_(mx::zeros({config.embed_dim})),
      norm2_weight_(mx::ones({config.embed_dim})),
      norm2_bias_(mx::zeros({config.embed_dim})),
      norm_eps_(config.layer_norm_eps)
{}

mx::array NomicTransformerBlock::operator()(const mx::array& inputs,
                                             const std::optional<mx::array>& mask) {
    auto attn_out = attention_(inputs, mask);
    auto add_norm = layer_norm(mx::add(attn_out, inputs), norm1_weight_, norm1_bias_, norm_eps_);
    auto mlp_out = mlp_(add_norm);
    return layer_norm(mx::add(add_norm, mlp_out), norm2_weight_, norm2_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> NomicTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["norm1.weight"] = &norm1_weight_;
    map["norm1.bias"] = &norm1_bias_;
    map["norm2.weight"] = &norm2_weight_;
    map["norm2.bias"] = &norm2_bias_;
    return map;
}

// --- NomicEncoder ---

NomicEncoder::NomicEncoder(const NomicBertConfiguration& config) {
    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i)
        layers_.emplace_back(config);
}

mx::array NomicEncoder::operator()(const mx::array& inputs,
                                     const std::optional<mx::array>& attention_mask) {
    auto output = inputs;
    for (auto& layer : layers_)
        output = layer(output, attention_mask);
    return output;
}

std::unordered_map<std::string, mx::array*> NomicEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- NomicLMHead ---

NomicLMHead::NomicLMHead(const NomicBertConfiguration& config)
    : dense_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      ln_weight_(mx::ones({config.embed_dim})),
      ln_bias_(mx::zeros({config.embed_dim})),
      decoder_weight_(mx::zeros({config.vocabulary_size, config.embed_dim})),
      norm_eps_(config.layer_norm_eps)
{
    if (config.mlp_fc1_bias) {
        dense_bias_ = mx::zeros({config.embed_dim});
        decoder_bias_ = mx::zeros({config.vocabulary_size});
    }
}

mx::array NomicLMHead::operator()(const mx::array& inputs) {
    auto x = linear_fwd(inputs, dense_weight_, dense_bias_.has_value() ? &dense_bias_.value() : nullptr);
    x = silu(x);
    x = layer_norm(x, ln_weight_, ln_bias_, norm_eps_);
    return linear_fwd(x, decoder_weight_, decoder_bias_.has_value() ? &decoder_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> NomicLMHead::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["dense.weight"] = &dense_weight_;
    map["ln.weight"] = &ln_weight_;
    map["ln.bias"] = &ln_bias_;
    map["decoder.weight"] = &decoder_weight_;
    if (dense_bias_.has_value()) map["dense.bias"] = &dense_bias_.value();
    if (decoder_bias_.has_value()) map["decoder.bias"] = &decoder_bias_.value();
    return map;
}

// --- NomicBertModel ---

NomicBertModel::NomicBertModel(const NomicBertConfiguration& config,
                               bool pooler, bool lm_head)
    : config_(config),
      embedder_(config),
      encoder_(config),
      vocabulary_size_(config.vocabulary_size)
{
    if (pooler) {
        pooler_weight_ = mx::zeros({config.embed_dim, config.embed_dim});
    }
    if (lm_head) {
        lm_head_.emplace(config);
    }
}

EmbeddingModelOutput NomicBertModel::call_impl(
    const mx::array& inputs,
    const std::optional<mx::array>& position_ids,
    const std::optional<mx::array>& token_type_ids,
    const std::optional<mx::array>& attention_mask)
{
    auto inp = inputs;
    if (inp.ndim() == 1) {
        inp = mx::reshape(inp, {1, -1});
    }

    std::optional<mx::array> mask = attention_mask;
    if (mask.has_value()) {
        auto m = mx::astype(mask.value(), embedder_.word_embeddings_weight().dtype());
        m = mx::expand_dims(m, {1, 2});
        mask = mx::log(m);
    }

    auto outputs = encoder_(
        embedder_(inp, position_ids, token_type_ids),
        mask);

    if (lm_head_.has_value()) {
        return {lm_head_.value()(outputs), std::nullopt};
    }
    if (pooler_weight_.has_value()) {
        auto cls = mx::slice(outputs, {0, 0, 0}, {outputs.shape(0), 1, outputs.shape(2)});
        cls = mx::squeeze(cls, 1);
        auto pooled = mx::tanh(linear_fwd(cls, pooler_weight_.value()));
        return {outputs, pooled};
    }
    return {outputs, std::nullopt};
}

std::unordered_map<std::string, mx::array>
NomicBertModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    std::unordered_map<std::string, mx::array> result;
    for (auto& [key, val] : weights) {
        std::string k = key;
        auto replace = [](std::string& s, const std::string& from, const std::string& to) {
            size_t pos = 0;
            while ((pos = s.find(from, pos)) != std::string::npos) {
                s.replace(pos, from.length(), to);
                pos += to.length();
            }
        };
        replace(k, "emb_ln", "embeddings.norm");
        replace(k, "bert.", "");
        replace(k, "cls.predictions.transform.dense.", "lm_head.dense.");
        replace(k, "cls.predictions.transform.LayerNorm.", "lm_head.ln.");
        replace(k, "cls.predictions.decoder", "lm_head.decoder");
        replace(k, "pooler.dense.", "pooler.");
        result.insert_or_assign(k, val);
    }
    return result;
}

void NomicBertModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> NomicBertModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embedder_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    if (lm_head_.has_value()) {
        for (auto& [k, v] : lm_head_->weight_map()) map["lm_head." + k] = v;
    }
    if (pooler_weight_.has_value()) {
        map["pooler.weight"] = &pooler_weight_.value();
    }
    return map;
}

} // namespace mlx_lm

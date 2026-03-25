// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/embedders/bert.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <algorithm>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, BertConfiguration& c) {
    c.model_type = j.at("model_type").get<std::string>();
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-12f);
    c.vocabulary_size = j.value("vocab_size", 30528);
    c.max_position_embeddings = j.value("max_position_embeddings", 0);

    if (c.model_type == "distilbert") {
        c.embed_dim = j.value("dim", 768);
        c.num_heads = j.value("n_heads", 12);
        c.inter_dim = j.value("hidden_dim", 3072);
        c.num_layers = j.value("n_layers", 12);
        c.type_vocabulary_size = 0;
    } else {
        c.embed_dim = j.value("hidden_size", 768);
        c.num_heads = j.value("num_attention_heads", 12);
        c.inter_dim = j.value("intermediate_size", 3072);
        c.num_layers = j.value("num_hidden_layers", 12);
        c.type_vocabulary_size = j.value("type_vocab_size", 2);
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* b = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (b) out = mx::add(out, *b);
    return out;
}

static mx::array layer_norm(const mx::array& x, const mx::array& w, const mx::array& b, float eps) {
    return mx::fast::layer_norm(x, w, b, eps);
}

// --- BertEmbedding ---

BertEmbedding::BertEmbedding(const BertConfiguration& config)
    : word_embeddings_weight_(mx::zeros({config.vocabulary_size, config.embed_dim})),
      norm_weight_(mx::ones({config.embed_dim})),
      norm_bias_(mx::zeros({config.embed_dim})),
      norm_eps_(config.layer_norm_eps),
      position_embeddings_weight_(mx::zeros({
          config.max_position_embeddings > 0 ? config.max_position_embeddings : 512,
          config.embed_dim})),
      type_vocabulary_size_(config.type_vocabulary_size)
{
    if (config.type_vocabulary_size > 0) {
        token_type_embeddings_weight_ = mx::zeros({config.type_vocabulary_size, config.embed_dim});
    }
}

mx::array BertEmbedding::operator()(const mx::array& input_ids,
                                     const std::optional<mx::array>& position_ids,
                                     const std::optional<mx::array>& token_type_ids) {
    int seq_len = input_ids.shape(1);
    auto pos_ids = position_ids.has_value() ? position_ids.value()
        : mx::broadcast_to(mx::arange(seq_len), input_ids.shape());

    auto words = mx::add(
        mx::take(word_embeddings_weight_, input_ids, 0),
        mx::take(position_embeddings_weight_, pos_ids, 0));

    if (token_type_ids.has_value() && token_type_embeddings_weight_.has_value()) {
        words = mx::add(words, mx::take(token_type_embeddings_weight_.value(), token_type_ids.value(), 0));
    }

    return layer_norm(words, norm_weight_, norm_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> BertEmbedding::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["word_embeddings.weight"] = &word_embeddings_weight_;
    map["norm.weight"] = &norm_weight_;
    map["norm.bias"] = &norm_bias_;
    map["position_embeddings.weight"] = &position_embeddings_weight_;
    if (token_type_embeddings_weight_.has_value()) {
        map["token_type_embeddings.weight"] = &token_type_embeddings_weight_.value();
    }
    return map;
}

// --- BertTransformerBlock ---

BertTransformerBlock::BertTransformerBlock(const BertConfiguration& config)
    : num_heads_(config.num_heads),
      head_dim_(config.embed_dim / config.num_heads),
      norm_eps_(config.layer_norm_eps),
      query_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      query_bias_(mx::zeros({config.embed_dim})),
      key_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      key_bias_(mx::zeros({config.embed_dim})),
      value_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      value_bias_(mx::zeros({config.embed_dim})),
      out_proj_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      out_proj_bias_(mx::zeros({config.embed_dim})),
      ln1_weight_(mx::ones({config.embed_dim})),
      ln1_bias_(mx::zeros({config.embed_dim})),
      ln2_weight_(mx::ones({config.embed_dim})),
      ln2_bias_(mx::zeros({config.embed_dim})),
      linear1_weight_(mx::zeros({config.inter_dim, config.embed_dim})),
      linear1_bias_(mx::zeros({config.inter_dim})),
      linear2_weight_(mx::zeros({config.embed_dim, config.inter_dim})),
      linear2_bias_(mx::zeros({config.embed_dim}))
{}

mx::array BertTransformerBlock::operator()(const mx::array& inputs,
                                            const std::optional<mx::array>& mask) {
    int B = inputs.shape(0), L = inputs.shape(1);

    auto q = linear_fwd(inputs, query_weight_, &query_bias_);
    auto k = linear_fwd(inputs, key_weight_, &key_bias_);
    auto v = linear_fwd(inputs, value_weight_, &value_bias_);

    q = mx::transpose(mx::reshape(q, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});

    auto output = mx::fast::scaled_dot_product_attention(
        q, k, v, std::pow(static_cast<float>(head_dim_), -0.5f), "", mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    auto attn_out = linear_fwd(output, out_proj_weight_, &out_proj_bias_);
    auto pre_norm = layer_norm(mx::add(inputs, attn_out), ln1_weight_, ln1_bias_, norm_eps_);

    auto mlp_out = linear_fwd(pre_norm, linear1_weight_, &linear1_bias_);
    // GELU activation
    mlp_out = mx::multiply(mlp_out,
        mx::multiply(mx::array(0.5f),
            mx::add(mx::array(1.0f), mx::erf(mx::divide(mlp_out, mx::array(std::sqrt(2.0f)))))));
    mlp_out = linear_fwd(mlp_out, linear2_weight_, &linear2_bias_);

    return layer_norm(mx::add(mlp_out, pre_norm), ln2_weight_, ln2_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> BertTransformerBlock::weight_map() {
    return {
        {"attention.query_proj.weight", &query_weight_},
        {"attention.query_proj.bias", &query_bias_},
        {"attention.key_proj.weight", &key_weight_},
        {"attention.key_proj.bias", &key_bias_},
        {"attention.value_proj.weight", &value_weight_},
        {"attention.value_proj.bias", &value_bias_},
        {"attention.out_proj.weight", &out_proj_weight_},
        {"attention.out_proj.bias", &out_proj_bias_},
        {"ln1.weight", &ln1_weight_}, {"ln1.bias", &ln1_bias_},
        {"ln2.weight", &ln2_weight_}, {"ln2.bias", &ln2_bias_},
        {"linear1.weight", &linear1_weight_}, {"linear1.bias", &linear1_bias_},
        {"linear2.weight", &linear2_weight_}, {"linear2.bias", &linear2_bias_},
    };
}

// --- BertEncoder ---

BertEncoder::BertEncoder(const BertConfiguration& config) {
    layers_.reserve(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i)
        layers_.emplace_back(config);
}

mx::array BertEncoder::operator()(const mx::array& inputs,
                                    const std::optional<mx::array>& attention_mask) {
    auto output = inputs;
    for (auto& layer : layers_) {
        output = layer(output, attention_mask);
    }
    return output;
}

std::unordered_map<std::string, mx::array*> BertEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- BertLMHead ---

BertLMHead::BertLMHead(const BertConfiguration& config)
    : dense_weight_(mx::zeros({config.embed_dim, config.embed_dim})),
      dense_bias_(mx::zeros({config.embed_dim})),
      ln_weight_(mx::ones({config.embed_dim})),
      ln_bias_(mx::zeros({config.embed_dim})),
      norm_eps_(config.layer_norm_eps),
      decoder_weight_(mx::zeros({config.vocabulary_size, config.embed_dim})),
      decoder_bias_(mx::zeros({config.vocabulary_size}))
{}

mx::array BertLMHead::operator()(const mx::array& inputs) {
    auto x = linear_fwd(inputs, dense_weight_, &dense_bias_);
    x = silu(x);
    x = layer_norm(x, ln_weight_, ln_bias_, norm_eps_);
    return linear_fwd(x, decoder_weight_, &decoder_bias_);
}

std::unordered_map<std::string, mx::array*> BertLMHead::weight_map() {
    return {
        {"dense.weight", &dense_weight_}, {"dense.bias", &dense_bias_},
        {"ln.weight", &ln_weight_}, {"ln.bias", &ln_bias_},
        {"decoder.weight", &decoder_weight_}, {"decoder.bias", &decoder_bias_},
    };
}

// --- BertModel ---

BertModel::BertModel(const BertConfiguration& config, bool lm_head)
    : config_(config),
      embedder_(config),
      encoder_(config),
      vocabulary_size_(config.vocabulary_size)
{
    if (lm_head) {
        lm_head_.emplace(config);
    } else {
        pooler_weight_ = mx::zeros({config.embed_dim, config.embed_dim});
        pooler_bias_ = mx::zeros({config.embed_dim});
    }
}

EmbeddingModelOutput BertModel::call_impl(
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
    } else {
        // CLS pooling
        auto cls = mx::slice(outputs, {0, 0, 0}, {outputs.shape(0), 1, outputs.shape(2)});
        cls = mx::squeeze(cls, 1);
        auto pooled = mx::tanh(linear_fwd(cls, pooler_weight_.value(), &pooler_bias_.value()));
        return {outputs, pooled};
    }
}

std::unordered_map<std::string, mx::array>
BertModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    std::unordered_map<std::string, mx::array> result;
    bool is_distilbert = (config_.model_type == "distilbert");

    for (auto& [key, val] : weights) {
        if (key == "embeddings.position_ids") continue;

        std::string k = key;
        if (is_distilbert) {
            // DistilBERT key mappings
            auto replace = [](std::string& s, const std::string& from, const std::string& to) {
                size_t pos = 0;
                while ((pos = s.find(from, pos)) != std::string::npos) {
                    s.replace(pos, from.length(), to);
                    pos += to.length();
                }
            };
            replace(k, ".layer.", ".layers.");
            replace(k, "transformer.", "encoder.");
            replace(k, "embeddings.LayerNorm", "embeddings.norm");
            replace(k, ".attention.q_lin.", ".attention.query_proj.");
            replace(k, ".attention.k_lin.", ".attention.key_proj.");
            replace(k, ".attention.v_lin.", ".attention.value_proj.");
            replace(k, ".attention.out_lin.", ".attention.out_proj.");
            replace(k, ".sa_layer_norm.", ".ln1.");
            replace(k, ".ffn.lin1.", ".linear1.");
            replace(k, ".ffn.lin2.", ".linear2.");
            replace(k, ".output_layer_norm.", ".ln2.");
            replace(k, "vocab_transform", "lm_head.dense");
            replace(k, "vocab_layer_norm", "lm_head.ln");
            replace(k, "vocab_projector", "lm_head.decoder");
            replace(k, "distilbert.", "");
        } else {
            // BERT key mappings
            auto replace = [](std::string& s, const std::string& from, const std::string& to) {
                size_t pos = 0;
                while ((pos = s.find(from, pos)) != std::string::npos) {
                    s.replace(pos, from.length(), to);
                    pos += to.length();
                }
            };
            replace(k, ".layer.", ".layers.");
            replace(k, ".self.key.", ".key_proj.");
            replace(k, ".self.query.", ".query_proj.");
            replace(k, ".self.value.", ".value_proj.");
            replace(k, ".attention.output.dense.", ".attention.out_proj.");
            replace(k, ".attention.output.LayerNorm.", ".ln1.");
            replace(k, ".output.LayerNorm.", ".ln2.");
            replace(k, ".intermediate.dense.", ".linear1.");
            replace(k, ".output.dense.", ".linear2.");
            replace(k, ".LayerNorm.", ".norm.");
            replace(k, "pooler.dense.", "pooler.");
            replace(k, "cls.predictions.transform.dense.", "lm_head.dense.");
            replace(k, "cls.predictions.transform.LayerNorm.", "lm_head.ln.");
            replace(k, "cls.predictions.decoder", "lm_head.decoder");
            replace(k, "cls.predictions.transform.norm.weight", "lm_head.ln.weight");
            replace(k, "cls.predictions.transform.norm.bias", "lm_head.ln.bias");
            replace(k, "cls.predictions.bias", "lm_head.decoder.bias");
            replace(k, "bert.", "");
        }
        result.insert_or_assign(k, val);
    }
    return result;
}

void BertModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> BertModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embedder_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    if (lm_head_.has_value()) {
        for (auto& [k, v] : lm_head_->weight_map()) map["lm_head." + k] = v;
    }
    if (pooler_weight_.has_value()) {
        map["pooler.weight"] = &pooler_weight_.value();
        map["pooler.bias"] = &pooler_bias_.value();
    }
    return map;
}

} // namespace mlx_lm

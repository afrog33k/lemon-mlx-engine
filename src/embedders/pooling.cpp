// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/embedders/pooling.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, PoolingConfiguration& c) {
    c.dimension = j.value("word_embedding_dimension", 0);
    c.pooling_mode_cls_token = j.value("pooling_mode_cls_token", false);
    c.pooling_mode_mean_tokens = j.value("pooling_mode_mean_tokens", false);
    c.pooling_mode_max_tokens = j.value("pooling_mode_max_tokens", false);
    c.pooling_mode_last_token = j.value("pooling_mode_lasttoken", false);
}

Pooling::Pooling(const PoolingConfiguration& config) {
    dimension_ = config.dimension > 0 ? std::optional<int>(config.dimension) : std::nullopt;
    if (config.pooling_mode_cls_token) {
        strategy_ = Strategy::CLS;
    } else if (config.pooling_mode_mean_tokens) {
        strategy_ = Strategy::Mean;
    } else if (config.pooling_mode_max_tokens) {
        strategy_ = Strategy::Max;
    } else if (config.pooling_mode_last_token) {
        strategy_ = Strategy::Last;
    } else {
        strategy_ = Strategy::First;
    }
}

mx::array Pooling::operator()(
    const EmbeddingModelOutput& inputs,
    const std::optional<mx::array>& mask,
    bool normalize,
    bool apply_layer_norm) const
{
    auto hidden = inputs.hidden_states.value();
    auto _mask = mask.has_value() ? mask.value()
        : mx::ones({hidden.shape(0), hidden.shape(1)});

    mx::array pooled(0.0f);
    switch (strategy_) {
    case Strategy::Mean: {
        auto expanded_mask = mx::expand_dims(_mask, {-1});
        auto masked = mx::multiply(hidden, expanded_mask);
        auto summed = mx::sum(masked, 1);
        auto mask_sum = mx::sum(_mask, -1, true);
        pooled = mx::divide(summed, mask_sum);
        break;
    }
    case Strategy::Max: {
        auto expanded_mask = mx::expand_dims(_mask, {-1});
        auto masked = mx::multiply(hidden, expanded_mask);
        pooled = mx::max(masked, 1);
        break;
    }
    case Strategy::First:
        pooled = mx::slice(hidden, {0, 0, 0}, {hidden.shape(0), 1, hidden.shape(2)});
        pooled = mx::squeeze(pooled, 1);
        break;
    case Strategy::Last:
        pooled = mx::slice(hidden, {0, hidden.shape(1) - 1, 0},
                           {hidden.shape(0), hidden.shape(1), hidden.shape(2)});
        pooled = mx::squeeze(pooled, 1);
        break;
    case Strategy::CLS:
        if (inputs.pooled_output.has_value()) {
            pooled = inputs.pooled_output.value();
        } else {
            pooled = mx::slice(hidden, {0, 0, 0}, {hidden.shape(0), 1, hidden.shape(2)});
            pooled = mx::squeeze(pooled, 1);
        }
        break;
    case Strategy::None:
        pooled = inputs.pooled_output.has_value() ? inputs.pooled_output.value() : hidden;
        break;
    }

    if (apply_layer_norm) {
        pooled = mx::fast::layer_norm(pooled, std::nullopt, std::nullopt, 1e-5f);
    }

    if (dimension_.has_value()) {
        int dim = dimension_.value();
        pooled = mx::slice(pooled, {0, 0}, {pooled.shape(0), dim});
    }

    if (normalize) {
        auto n = mx::sqrt(mx::sum(mx::square(pooled), -1, true));
        pooled = mx::divide(pooled, n);
    }

    return pooled;
}

Pooling load_pooling(const std::string& model_directory) {
    auto config_path = fs::path(model_directory) / "1_Pooling" / "config.json";
    if (!fs::exists(config_path)) {
        return Pooling(Pooling::Strategy::None);
    }

    std::ifstream f(config_path);
    nlohmann::json j;
    f >> j;

    PoolingConfiguration config = j.get<PoolingConfiguration>();
    return Pooling(config);
}

} // namespace mlx_lm

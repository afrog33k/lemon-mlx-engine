// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/embedders/embedding_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace mlx_lm {

struct PoolingConfiguration {
    int dimension = 0;
    bool pooling_mode_cls_token = false;
    bool pooling_mode_mean_tokens = false;
    bool pooling_mode_max_tokens = false;
    bool pooling_mode_last_token = false;
};

void from_json(const nlohmann::json& j, PoolingConfiguration& c);

class Pooling {
public:
    enum class Strategy {
        Mean, Max, CLS, First, Last, None
    };

    explicit Pooling(Strategy strategy = Strategy::None, std::optional<int> dimension = std::nullopt)
        : strategy_(strategy), dimension_(dimension) {}

    explicit Pooling(const PoolingConfiguration& config);

    mlx::core::array operator()(
        const EmbeddingModelOutput& inputs,
        const std::optional<mlx::core::array>& mask = std::nullopt,
        bool normalize = false,
        bool apply_layer_norm = false) const;

    Strategy strategy() const { return strategy_; }

private:
    Strategy strategy_ = Strategy::None;
    std::optional<int> dimension_;
};

Pooling load_pooling(const std::string& model_directory);

} // namespace mlx_lm

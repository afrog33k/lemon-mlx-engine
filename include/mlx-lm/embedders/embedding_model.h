// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

struct EmbeddingModelOutput {
    std::optional<mlx::core::array> hidden_states;
    std::optional<mlx::core::array> pooled_output;
};

template <typename Derived>
class EmbeddingModel {
public:
    EmbeddingModelOutput operator()(
        const mlx::core::array& inputs,
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& token_type_ids = std::nullopt,
        const std::optional<mlx::core::array>& attention_mask = std::nullopt)
    {
        return derived().call_impl(inputs, position_ids, token_type_ids, attention_mask);
    }

    std::unordered_map<std::string, mlx::core::array>
    sanitize(std::unordered_map<std::string, mlx::core::array> weights) {
        return derived().sanitize_impl(std::move(weights));
    }

    int vocab_size() const { return derived().vocab_size_impl(); }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

} // namespace mlx_lm

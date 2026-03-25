// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of RoPEUtils.swift and SuScaledRoPE.swift
#pragma once

#include <mlx-lm/common/string_utils.h>
#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx_lm {

// ---------------------------------------------------------------------------
// Llama3RoPE — Llama-3 style RoPE with per-frequency scaling.
//
// Uses factor / low_freq_factor / high_freq_factor / original_max_position_embeddings
// from the rope_scaling config to compute custom per-dimension frequencies,
// then passes them to mx::fast::rope via the freqs parameter.
// ---------------------------------------------------------------------------
class Llama3RoPE {
    int dims_;
    int max_position_embeddings_;
    bool traditional_;
    mlx::core::array freqs_;

public:
    Llama3RoPE(
        int dims,
        int max_position_embeddings = 2048,
        bool traditional = false,
        float base = 10000.0f,
        const std::optional<std::unordered_map<std::string, StringOrNumber>>& scaling_config = std::nullopt);

    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

// ---------------------------------------------------------------------------
// YarnRoPE — Yet Another RoPE Naturally (YaRN) rotary position embedding.
//
// Combines interpolation and extrapolation with a linear ramp mask to smoothly
// transition between frequency ranges. Includes mscale correction for
// scaling factor.
// ---------------------------------------------------------------------------
class YarnRoPE {
    int dims_;
    bool traditional_;
    int max_position_embeddings_;
    float base_;
    float scaling_factor_;
    int original_max_position_embeddings_;
    float beta_fast_;
    float beta_slow_;
    float mscale_;
    float mscale_all_dim_;

    float computed_mscale_;
    mlx::core::array freqs_;

public:
    YarnRoPE(
        int dims,
        bool traditional = false,
        int max_position_embeddings = 2048,
        float base = 10000.0f,
        float scaling_factor = 1.0f,
        int original_max_position_embeddings = 4096,
        float beta_fast = 32.0f,
        float beta_slow = 1.0f,
        float mscale = 1.0f,
        float mscale_all_dim = 0.0f);

    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

// ---------------------------------------------------------------------------
// SuScaledRoPE — Su Scaled Rotary Position Embedding (longrope).
//
// Switches between short and long factor frequencies based on sequence length
// relative to original_max_position_embeddings.
// ---------------------------------------------------------------------------
class SuScaledRoPE {
    int dims_;
    int original_max_position_embeddings_;
    mlx::core::array short_freqs_;
    mlx::core::array long_freqs_;
    float short_scale_;
    float long_scale_;

public:
    SuScaledRoPE(
        int dims,
        float base = 10000.0f,
        int max_position_embeddings = 131072,
        int original_max_position_embeddings = 4096,
        const std::vector<float>& short_factor = {1.0f},
        const std::vector<float>& long_factor = {1.0f},
        std::optional<float> short_m_scale = std::nullopt,
        std::optional<float> long_m_scale = std::nullopt);

    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

// ---------------------------------------------------------------------------
// RoPEVariant — type-erased RoPE that can hold any of the above variants
// or a simple (base, scale) configuration for the default/linear RoPE.
// ---------------------------------------------------------------------------

// SimpleRoPE just stores params for mx::fast::rope (default/linear/mrope).
struct SimpleRoPE {
    int dims;
    bool traditional;
    float base;
    float scale;

    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

// Type-erased RoPE variant.
using RoPEVariant = std::variant<SimpleRoPE, Llama3RoPE, YarnRoPE, SuScaledRoPE>;

// Apply a RoPEVariant to an input tensor.
mlx::core::array apply_rope(RoPEVariant& rope, const mlx::core::array& x, int offset = 0);

// ---------------------------------------------------------------------------
// initializeRope — factory that creates the right RoPE variant from config.
//
// This mirrors the Swift initializeRope() function. The rope_scaling_json
// parameter is the raw JSON object for extracting array values (short_factor,
// long_factor) that the StringOrNumber map cannot represent.
// ---------------------------------------------------------------------------
RoPEVariant initialize_rope(
    int dims,
    float base,
    bool traditional,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& scaling_config = std::nullopt,
    std::optional<int> max_position_embeddings = std::nullopt,
    const nlohmann::json* rope_scaling_json = nullptr);

} // namespace mlx_lm

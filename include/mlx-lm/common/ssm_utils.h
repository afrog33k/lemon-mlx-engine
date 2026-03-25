// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SSM.swift — Mamba2 SSM step functions
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <utility>

namespace mlx_lm {

// Compute dt with bias and clamping: softplus(dt + dt_bias), then clip.
mlx::core::array compute_dt(
    const mlx::core::array& dt,
    const mlx::core::array& dt_bias,
    float time_step_min = 0.0f,
    float time_step_max = 1e30f);

// Segment sum helper for ssmAttn path.
mlx::core::array segsum(
    const mlx::core::array& x,
    const std::optional<mlx::core::array>& mask = std::nullopt);

// Mamba2 SSM update using attention-like formulation (for prefill, seqLen > 1).
// x: [B, L, H, Dh], B_ssm: [B, L, G, Ds], C_ssm: [B, L, G, Ds]
// dt: [B, L, H], A_log: [H], D: [H], dt_bias: [H]
// Returns (y: [B, L, H, Dh], new_state).
std::pair<mlx::core::array, mlx::core::array> ssm_attn(
    const mlx::core::array& x,
    const mlx::core::array& A_log,
    const mlx::core::array& B_ssm,
    const mlx::core::array& C_ssm,
    const mlx::core::array& D,
    const mlx::core::array& dt,
    const mlx::core::array& dt_bias,
    const std::optional<mlx::core::array>& state = std::nullopt,
    float time_step_min = 0.001f,
    float time_step_max = 100.0f,
    const std::optional<mlx::core::array>& mask = std::nullopt);

// Mamba2 SSM update — dispatches to ssm_attn (pure MLX ops, no Metal kernel).
// Same interface as Swift ssmUpdate().
std::pair<mlx::core::array, mlx::core::array> ssm_update(
    const mlx::core::array& hidden_states,
    const mlx::core::array& A_log,
    const mlx::core::array& B_ssm,
    const mlx::core::array& C_ssm,
    const mlx::core::array& D,
    const mlx::core::array& dt,
    const mlx::core::array& dt_bias,
    const std::optional<mlx::core::array>& state = std::nullopt,
    float time_step_min = 0.001f,
    float time_step_max = 100.0f,
    const std::optional<mlx::core::array>& mask = std::nullopt);

} // namespace mlx_lm

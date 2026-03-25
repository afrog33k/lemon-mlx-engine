// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SwitchLayers.swift — MoE routing via gather_mm
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace mlx_lm {

// Sort indices for efficient gather_mm dispatch.
// Returns (sorted_x, sorted_indices, inverse_order).
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
gather_sort(const mlx::core::array& x, const mlx::core::array& indices);

// Unsort results back to original order.
mlx::core::array scatter_unsort(
    const mlx::core::array& x,
    const mlx::core::array& inv_order,
    const mlx::core::Shape* shape = nullptr);

// SwitchLinear — a linear layer with multiple expert weight matrices.
// Uses gather_mm for efficient expert dispatch.
class SwitchLinear {
    mlx::core::array weight_;  // [num_experts, output_dims, input_dims]
    std::optional<mlx::core::array> bias_;  // [num_experts, output_dims] or nullopt

    int input_dims_;
    int output_dims_;
    int num_experts_;

public:
    SwitchLinear(int input_dims, int output_dims, int num_experts, bool bias = false);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& indices,
        bool sorted_indices = false);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// SwitchGLU — gated linear unit with expert routing.
// Applies gate_proj, up_proj, down_proj via SwitchLinear with silu activation.
class SwitchGLU {
    SwitchLinear gate_proj_;
    SwitchLinear up_proj_;
    SwitchLinear down_proj_;

    int input_dims_;
    int hidden_dims_;
    int num_experts_;

public:
    SwitchGLU(int input_dims, int hidden_dims, int num_experts, bool bias = false);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& indices);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm

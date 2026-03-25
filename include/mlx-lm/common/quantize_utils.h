// Copyright © 2025 — Ported to C++
#pragma once

#include <mlx-lm/common/base_config.h>
#include <mlx/mlx.h>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Register quantized weights in the QuantizedWeightRegistry.
//
// For each key ending in ".scales" with a matching ".weight", registers
// the quantization metadata (scales, biases, group_size, bits) so that
// linear_fwd() from quantized_linear.h will use mx::quantized_matmul.
//
// Requires the model's weight_map so we can map weight names to member
// array addresses (which linear_fwd uses for registry lookups).
//
// Removes .scales and .biases entries from the weights map after
// registration (the packed uint32 .weight entries stay).
void register_quantized_weights(
    std::unordered_map<std::string, mlx::core::array>& weights,
    const BaseConfiguration& config,
    const std::unordered_map<std::string, mlx::core::array*>& weight_map);

// Legacy: dequantize weights at load time (uses more memory).
// Kept for models that haven't been updated to use quantized_linear.h yet.
std::unordered_map<std::string, mlx::core::array> dequantize_weights(
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& config);

} // namespace mlx_lm

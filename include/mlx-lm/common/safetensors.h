// Copyright © 2025 — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// Load weights from a single .safetensors file.
std::unordered_map<std::string, mlx::core::array>
load_safetensors(const std::string& path);

// Load weights from a directory, handling sharded models
// (model-00001-of-00005.safetensors, etc.).
std::unordered_map<std::string, mlx::core::array>
load_safetensors_from_directory(const std::string& directory);

// Load model weights and apply them to a weight map.
// The weight_map maps parameter names to pointers where weights should be stored.
void load_weights(
    const std::string& model_directory,
    std::unordered_map<std::string, mlx::core::array*>& weight_map);

} // namespace mlx_lm

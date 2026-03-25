// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/registry.h>
#include <functional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// VLM type registry — maps model_type strings to factory functions.
ModelTypeRegistry& vlm_type_registry();

// VLM model registry — known VLM model configurations.
AbstractModelRegistry& vlm_model_registry();

// Load a VLM model from a local directory.
ModelContext load_vlm_from_directory(
    const std::string& model_directory,
    const ModelConfiguration& config = {});

// Load a VLM model from a Hugging Face model ID.
ModelContext load_vlm(
    const std::string& model_id,
    const std::string& cache_dir = "");

} // namespace mlx_lm

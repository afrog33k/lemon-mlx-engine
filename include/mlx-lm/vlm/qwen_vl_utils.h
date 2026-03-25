// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Shared utilities for Qwen VL models (Qwen2VL, Qwen2.5VL, Qwen3VL)
// Port of QwenVL.swift
#pragma once

#include <mlx-lm/common/types.h>
#include <mlx/mlx.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mlx_lm {
namespace qwen_vl {

// Rotates half the hidden dims of the input.
// x[..., :half] and x[..., half:] are swapped with negation.
mlx::core::array rotate_half(const mlx::core::array& x);

// Merge image/video features into text embeddings at positions of special tokens.
// inputIds: [L] flat token ids
// inputEmbeds: [L, D] or [1, L, D] text embeddings
// imageFeatures: [N, D] or [1, N, D] vision features
// Returns: [1, L, D] merged embeddings
mlx::core::array merge_input_ids_with_image_features(
    const mlx::core::array& input_ids,
    const mlx::core::array& input_embeds,
    const mlx::core::array& image_features,
    int image_token_id,
    int video_token_id);

// Vision Rotary Embedding — computes RoPE frequencies for vision transformer.
class VisionRotaryEmbedding {
    int dimensions_;
    float theta_;
    mlx::core::array inverse_freq_;

public:
    VisionRotaryEmbedding(int dimensions, float theta = 10000.0f);

    // Compute frequency embeddings for a given sequence length.
    // Returns: [seq_len, dimensions/2]
    mlx::core::array operator()(int sequence_length) const;
};

// Smart resize: compute target (height, width) preserving aspect ratio,
// constrained by factor alignment, min/max pixel counts.
// Port of image_processing_qwen2_vl.smart_resize
std::pair<int, int> target_size(
    int height, int width, int factor, int min_pixels, int max_pixels);

// Convert a sequence of image arrays into patches for the vision transformer.
// images: vector of [C, H, W] arrays (all same H, W)
// Returns: (flattened_patches [T*gridH*gridW, C*tps*ps*ps], THW grid)
std::pair<mlx::core::array, THW> patchify(
    const std::vector<mlx::core::array>& images,
    int merge_size, int patch_size, int temporal_patch_size);

} // namespace qwen_vl
} // namespace mlx_lm

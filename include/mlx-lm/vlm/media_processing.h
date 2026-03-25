// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Cross-platform replacement for MediaProcessing.swift
// Uses stb_image instead of CoreImage/AVFoundation
#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>

namespace mlx_lm {

// Cross-platform image processing utilities for VLM models.
// Replaces CoreImage-based MediaProcessing from Swift.
namespace media_processing {

// Load an image from file and return as MLXArray [1, C, H, W] in float32.
// Uses stb_image for cross-platform image loading.
mlx::core::array load_image(const std::string& path);

// Resize an image array [1, C, H, W] to target (height, width) using bilinear interpolation.
mlx::core::array resize_image(const mlx::core::array& image, int target_height, int target_width);

// Normalize image channels: output[c] = (input[c] - mean[c]) / std[c]
// image shape: [1, C, H, W] or [C, H, W]
mlx::core::array normalize(
    const mlx::core::array& image,
    const std::vector<float>& mean,
    const std::vector<float>& std_dev);

// Center crop an image [1, C, H, W] to target (height, width).
mlx::core::array center_crop(const mlx::core::array& image, int target_height, int target_width);

// Pad image to square by adding black borders.
mlx::core::array pad_to_square(const mlx::core::array& image);

// Rescale pixel values from [0, 255] to [0.0, 1.0].
mlx::core::array rescale(const mlx::core::array& image, float scale = 1.0f / 255.0f);

// Load image from raw pixel data (for in-memory images).
// data: raw RGB bytes, width x height x channels
mlx::core::array from_raw_pixels(
    const uint8_t* data, int width, int height, int channels = 3);

} // namespace media_processing
} // namespace mlx_lm
